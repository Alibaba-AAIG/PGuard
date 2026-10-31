# Gate-first top-k layer selection + multi-scale causal EMA + attention fusion.
# EMA consumes every real token; marker masks only select output positions.
# Gate scoring uses gate_norm-ed features before fusion, while the data path
# consumes raw features; training and inference must preserve this distinction.

import math
import os

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput, BertLayer

from src.utils.model_utils import BasicClassificationHead, config_from_dict
from src.utils.modeling_outputs import PGuardMultiTasksOutputForCls


class FrozenCausalEMA(nn.Module):
    """Fixed multi-scale EMA parameterized by effective decay lengths."""
    def __init__(self, hidden_size, effective_lengths=[25, 150]):
        super().__init__()
        self.n_heads = len(effective_lengths)
        self.hidden_size = hidden_size
        # alpha = 1 - exp(-1/tau), tau being the effective length
        init_alphas = [1.0 - math.exp(-1.0 / tau) for tau in effective_lengths]
        # Keep alphas as a plain attribute: DDP broadcasts registered buffers at
        # every forward, which can deadlock if another rank skips that forward.
        # These fixed values are identical on every rank and need no synchronization.
        self.alphas = torch.tensor([
            [a] * hidden_size for a in init_alphas
        ])  # (n_heads, hidden_size)

    def forward(self, x, padding_mask=None, sent_mask=None):
        # x: (B, S, H), padding_mask: (B, S), sent_mask: (S,) boolean
        # sent_mask: when provided, only EMA outputs at sent_mask[i]=True positions are stored, avoiding a full (B, S, H) tensor allocation
        B, S, H = x.shape
        all_ema = []
        for head_idx in range(self.n_heads):
            alpha = self.alphas[head_idx].to(x.device)  # (H,) a plain attribute does not follow model.to(); sync the device manually
            ema = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            outputs = []
            for i in range(S):
                xi = x[:, i, :]  # (B, H)
                if padding_mask is not None:
                    xi = xi * padding_mask[:, i].unsqueeze(-1).to(xi.dtype)
                ema = alpha * xi + (1 - alpha) * ema
                if sent_mask is None or sent_mask[i]:
                    outputs.append(ema.unsqueeze(1))
            if len(outputs) > 0:
                all_ema.append(torch.cat(outputs, dim=1))  # (B, S', H) or (B, S, H)
            else:
                all_ema.append(torch.zeros(B, 0, H, device=x.device, dtype=x.dtype))
        return all_ema  # list of (B, S', H)


class AttentionEMAFusionLayer(nn.Module):
    """Fuse [h_ema_0, ..., h_ema_{n-1}, x_t] with causal BERT attention.

    Scales are sorted from longest to shortest, so information flows toward x_t;
    the x_t position provides the fused output.
    """
    def __init__(self, hidden_size, effective_lengths=[25, 150], n_bert_layers=2, n_attn_heads=8):
        super().__init__()
        # descending order so that for i<j, h_ema_i carries no less history than h_ema_j
        effective_lengths = sorted(effective_lengths, reverse=True)
        self.ema_module = FrozenCausalEMA(hidden_size, effective_lengths)
        self.n_ema_heads = self.ema_module.n_heads
        self.seq_len = self.n_ema_heads + 1  # n EMA tokens + 1 x_t token
        self.hidden_size = hidden_size

        # BERT layers for attention fusion
        bert_config = BertConfig()
        bert_config.hidden_size = hidden_size
        bert_config.intermediate_size = hidden_size * 4
        bert_config.add_cross_attention = False
        bert_config.is_decoder = False
        bert_config.num_attention_heads = n_attn_heads
        self.bert_config = bert_config  # fuse() switches _attn_implementation on demand
        self.bert_layers = nn.ModuleList([
            BertLayer(bert_config) for _ in range(n_bert_layers)
        ])

    def compute_ema(self, features, padding_mask=None, sent_mask=None):
        """Compute EMA over full features (B, S, H).

        sent_mask (S,) filters stored outputs, not the recurrence history.
        Returns one (B, S', H) tensor per EMA scale.
        """
        ema_heads = self.ema_module(features, padding_mask, sent_mask=sent_mask)  # list of (B, S', H)
        return ema_heads

    def fuse(self, ema_heads, features, output_attentions=False):
        """Fuse filtered ema_heads and features, both with shape (B, S', H).

        Returns fused (B, S', H), plus per-layer attention weights of shape
        (B*S', n_heads, n+1, n+1) when output_attentions=True.
        """
        B, S, H = features.shape

        # 1. stack [h_ema_0, ..., h_ema_{n-1}, x_t] -> (B*S', n+1, H)
        tokens = ema_heads + [features]  # list of (B, S', H), len=n_ema_heads+1
        stacked = torch.stack(tokens, dim=2)  # (B, S', n+1, H)
        stacked = stacked.reshape(B * S, self.seq_len, H)  # (B*S', n+1, H)

        # 2. standard causal attention mask (lower triangular)
        #    information flow: long-scale h_ema -> short-scale h_ema -> x_t
        #    format: (1, 1, seq_len, seq_len) float, 0=attend, -inf=masked
        attn_mask = torch.triu(
            torch.full((self.seq_len, self.seq_len), torch.finfo(features.dtype).min,
                       device=features.device, dtype=features.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 3. BertLayers one by one (self-attention + FFN, with residual and LayerNorm)
        # transformers 5.2.0: BertLayer.forward discards attention weights; a
        # forward_hook on BertAttention is needed to capture them (same mechanism
        # as HuggingFace's official capture_outputs decorator)
        all_attentions = []
        hooks = []
        if output_attentions:
            # switch to eager only when attention weights are needed (sdpa returns no attention)
            orig_attn_impl = self.bert_config._attn_implementation
            self.bert_config._attn_implementation = "eager"
            def capture_attn_hook(module, input, output):
                # output = (attention_output, attn_weights) from BertAttention
                attn_weights = output[1]
                if attn_weights is not None:
                    all_attentions.append(attn_weights)
            for bert_layer in self.bert_layers:
                h = bert_layer.attention.register_forward_hook(capture_attn_hook)
                hooks.append(h)

        for bert_layer in self.bert_layers:
            stacked = bert_layer(stacked, attention_mask=attn_mask, output_attentions=output_attentions)

        for h in hooks:
            h.remove()
        if output_attentions:
            self.bert_config._attn_implementation = orig_attn_impl

        # 4. take the output at the x_t position (the last one)
        fused = stacked[:, -1, :]  # (B*S', H)
        fused = fused.reshape(B, S, H)  # (B, S', H)

        if output_attentions:
            return fused, all_attentions
        return fused


class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output


class PluginGuard(nn.Module):

    def __init__(self, config):

        super().__init__()

        ### EMA Attention Fusion ###
        self.ema_fusion = AttentionEMAFusionLayer(config.hidden_size, effective_lengths=[200,150,100,50], n_bert_layers=1)

        ### gating (gate-first: topk layer selection + soft aggregation; scoring input passes a dedicated gate_norm first) ###
        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        bert_layer_config = BertConfig()
        bert_layer_config.add_cross_attention = False

        self.input_proj = nn.Linear(config.hidden_size, bert_layer_config.hidden_size)
        self.adapter = nn.ModuleList( [AdapterBlock(bert_layer_config) for _ in range(1)] )  # FFN only, no attention needed

        self.norm = nn.LayerNorm(config.hidden_size, eps=bert_layer_config.layer_norm_eps)
        # dedicated to gate scoring: one affine must not serve two distributions
        self.gate_norm = nn.LayerNorm(config.hidden_size, eps=bert_layer_config.layer_norm_eps)

        # build multi classification heads
        self.classifiers = nn.ModuleList()

        classification_heads = config.classifier["classification_heads"]
        for name, num_classes in classification_heads:
            # update config
            classifier_config = config_from_dict({
                "hidden_size": bert_layer_config.hidden_size,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes":  num_classes,
                "dropout_type": config.classifier["dropout_type"]
            })
            classifier = BasicClassificationHead(classifier_config)
            self.classifiers.add_module(name, classifier)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_gate_first(self, features):
        '''Score features (bs, layer, sent_len, hidden_size) before fusion.

        Returns topk_weights/topk_scores/topk_indices (B, topk, S) and
        all_scores (B, L, S); features and EMA reuse the same top-k weights.
        '''
        # gate_norm affects scoring only, never the downstream data path
        scores = self.gate(self.gate_norm(features)).squeeze(-1)   # (B, L, S)

        scores_no_noise = scores

        # Gumbel noise
        if self.training:
            U = torch.rand_like(scores)
            gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
            scores += gumbel_noise

        # temperature controls exploration strength
        scores /= self.gate_temperature

        # topk selection
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)  # (B, topk, S)

        topk_weights = F.softmax(topk_scores, dim=1)  # (B, topk, S)

        all_scores = F.softmax(scores_no_noise, dim=1)

        return topk_weights, topk_scores, topk_indices, all_scores

    def forward(
        self,
        features:torch.Tensor,  # (bs, layer, sent_len, hidden_size)
        token_type_ids:dict,    # dict(key=int, value=(bs, sent_len))
        label_type:str,
        label_marker:dict,
        output_attentions:bool=False,
        attention_mask:Optional[torch.Tensor]=None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], PGuardMultiTasksOutputForCls]:
        '''token_type_ids represents the marker_mask; values >0 mark tokens that contribute to the loss'''

        # ===== [1] EMA (computed over all captured layers) =====
        B, L, S, H = features.shape
        sent_mask = None
        if attention_mask is None or attention_mask.shape != (B, S):
            raise ValueError(
                "the EMA head requires the full attention_mask with shape (B, S); "
                "the wrapper must always pass the real token padding mask."
            )
        ema_padding_mask = (
            attention_mask.to(device=features.device, dtype=torch.bool)
            .unsqueeze(1).expand(B, L, S).reshape(B * L, S)
        )
        # the marker field of view only filters output positions; it must not
        # mask the EMA's history input.
        if token_type_ids is not None:
            sent_mask = torch.stack([mask for mask in token_type_ids.values()]).sum(0).sum(0) > 0  # (S,)
            # with no marker field of view at all (usually inference-side truncation),
            # sent_mask is all False: skip the filtering and keep the full sequence
            if not sent_mask.any():
                sent_mask = None

        # [1a] EMA computed on the full sequence + sent_mask filtering (merged into
        # one step to avoid allocating the full EMA tensor)
        features_flat = features.reshape(B * L, S, H)
        ema_heads = self.ema_fusion.compute_ema(features_flat, padding_mask=ema_padding_mask, sent_mask=sent_mask)  # list of (B*L, S', H)

        # [1b] sent_mask filtering (features, token_type_ids, input_ids)
        input_ids = kwargs.get('input_ids')  # (bs, FULL_sent_len) or None
        if sent_mask is not None:
            features = features[:, :, sent_mask, :]  # (B, L, S', H)
            token_type_ids = {id: mask[:, sent_mask] for id, mask in token_type_ids.items()}
            if input_ids is not None:
                input_ids = input_ids[:, sent_mask]

        # ===== [2] gate-first: topk layer selection before fusion =====
        # NOTE: the gate consumes gate_norm-ed features; norm of the data path happens after fusion
        S_prime = features.shape[2]
        gate_topk_weights, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate_first(features)  # (B, topk, S'), (B, topk, S'), (B, topk, S'), (B, L, S')
        n_scales = self.ema_fusion.n_ema_heads
        # merge the layer dim of features and EMA with the same set of topk weights
        expanded_indices = gate_topk_indices.unsqueeze(-1).expand(B, self.top_L, S_prime, H)
        topk_features = torch.gather(features, dim=1, index=expanded_indices)  # (B, topk, S', H)
        merged_cur = (topk_features * gate_topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, S', H)
        # ema_heads: list of (B*L, S', H) -> (B, L, n_scales, S', H)
        ema_stacked = torch.stack(ema_heads, dim=1).reshape(B, L, n_scales, S_prime, H)
        ema_expanded_indices = gate_topk_indices.unsqueeze(2).unsqueeze(-1).expand(B, self.top_L, n_scales, S_prime, H)
        topk_ema = torch.gather(ema_stacked, dim=1, index=ema_expanded_indices)  # (B, topk, n_scales, S', H)
        merged_ema = (topk_ema * gate_topk_weights.unsqueeze(2).unsqueeze(-1)).sum(dim=1)  # (B, n_scales, S', H)

        # ===== [3] single BERT attention fusion (M=B*S' instead of B*L*S') =====
        merged_ema_heads = [merged_ema[:, i] for i in range(n_scales)]  # list of (B, S', H)
        fused_result = self.ema_fusion.fuse(merged_ema_heads, merged_cur, output_attentions=output_attentions)
        fusion_attentions = None
        if output_attentions:
            features, raw_attentions = fused_result  # (B, S', H)
            # process: last layer -> x_t row -> mean over heads -> (B, S', n+1) (no L dim once the gate is first)
            if len(raw_attentions) > 0:
                last_attn = raw_attentions[-1]  # (B*S', n_heads, n+1, n+1)
                n_players = self.ema_fusion.seq_len
                fusion_attentions = last_attn[:, :, -1, :].mean(dim=1)  # (B*S', n+1)
                fusion_attentions = fusion_attentions.reshape(B, S_prime, n_players)  # (B, S', n+1)
            # under sdpa raw_attentions is empty and fusion_attentions stays None
        else:
            features = fused_result  # (B, S', H)

        # ===== [4] norm -> FFN -> cls =====
        # norm
        features = self.norm(features)

        # adapter
        features = self.input_proj(features)  # (bs, sent_len, inter_hidden_size)
        for layer in self.adapter:
            features = layer(features)  # (bs, sent_len, inter_hidden_size)

        # classifier
        full_logits_list = []
        for idx, classifier in enumerate(self.classifiers):
            logits = classifier(features)  # (bs, sent_len, num_classes)
            full_logits_list.append(logits)  # full logits

        # preparation
        attention_mask = None
        if token_type_ids is not None:
            bs, sent_len = features.shape[:2]
            max_marker = len(token_type_ids)
            n_head = len(full_logits_list)
            marker_ids = list(token_type_ids.keys())

            token_type_ids = torch.stack([token_type_ids[k] for k in marker_ids], 0)  # (max_marker, bs, sent_len)

            # attention mask: positions of all markers
            attention_mask = (token_type_ids.sum(0) > 0).to(torch.long)  # (bs,sent_len)

        # loss
        final_loss = None
        if token_type_ids is not None and label_type is not None and label_marker is not None:
            if label_type != 'single':
                raise NotImplementedError(label_type)

            # stack markers into the bs dimension
            token_type_ids = token_type_ids.flatten(0,1)                              # (bs', sent_len)

            logits_list = [x.unsqueeze(0).expand(max_marker, -1, -1, -1) for x in full_logits_list]  # list of (max_marker, bs, sent_len, num_classes), len=n_head
            logits_list = [x.flatten(0,1) for x in logits_list]                                      # list of (bs', sent_len, num_classes), len=n_head  # expand returns a view, but the following flatten copies and costs memory; num_classes is at most ~100 here, so acceptable

            labels = [{id: item['batch_single_labels'][i] for id, item in label_marker.items()} for i in range(n_head)]  # list of dict(key=int, value=(bs)), len=n_head
            labels = [torch.stack([x[k] for k in marker_ids], 0) for x in labels]     # list of (max_marker, bs), len=n_head
            labels = [x.flatten(0,1) for x in labels]                        # list of (bs'), len=n_head
            labels = [x.unsqueeze(-1).expand(-1, sent_len) for x in labels]  # list of (bs', sent_len), len=n_head  # the label of each (bs, marker) pair must apply to the tokens with token_type_ids>0, hence the extra sent_len dim

            task_mask = {id: item['batch_task_mask'] for id, item in label_marker.items()}  # dict(key=int, value=(bs,n_head))
            task_mask = torch.stack([task_mask[k] for k in marker_ids], 0)    # (max_marker, bs, n_head)
            task_mask = task_mask.flatten(0,1)                                # (bs', n_head)

            if (token_type_ids > 0).sum() == 0:
                _rank = os.environ.get('RANK', '?')
                print(f'[RANK={_rank}] zero-loss branch: no valid tokens remain in the marker fields of view; check truncation/max_length '
                      f'(valid_marker_tokens={int((token_type_ids > 0).sum().item())}, token_type_ids.shape={tuple(token_type_ids.shape)})', flush=True)
                # Keep the zero loss dependent on model parameters so this rank
                # still runs gradient allreduce and stays aligned with other ranks.
                final_loss = sum(x.sum() for x in full_logits_list) * 0.0
                print(f'[RANK={_rank}] zero-loss branch: using a parameter-dependent loss to keep DDP gradient allreduce aligned', flush=True)
            else:
                # if some (bs, marker) pair has no valid token (truncated), its task_mask
                # must be zeroed; data preprocessing does not enforce this, so handle it here
                # sequences without any valid token get task_mask forced to 0
                task_mask *= ((token_type_ids > 0).sum(1) > 0).float().unsqueeze(-1)

                multitask_loss = []
                for idx in range(n_head):
                    logits = logits_list[idx]   # (bs', sent_len, num_classes)
                    labels_ = labels[idx]       # (bs', sent_len)

                    loss = nn.CrossEntropyLoss(reduction="none")(logits.flatten(0,1), labels_.flatten(0,1)).view(-1, sent_len)  # (bs',sent_len)

                    loss *= (token_type_ids > 0).float()  # (bs', sent_len)
                    valid_token_counts = (token_type_ids > 0).float().sum(1).clamp(min=1e-8)  # (bs')
                    loss = loss.sum(1) / valid_token_counts  # (bs')

                    multitask_loss.append(loss)

                multitask_loss = torch.stack(multitask_loss, 1) # (bs', n_head)
                multitask_loss *= task_mask

                valid_task_counts = task_mask.float().sum().clamp(min=1e-8)
                final_loss = multitask_loss.sum() / valid_task_counts  # the final loss must not average over the bs dimension directly; task_mask counts the truly valid samples

            # restore shapes
            token_type_ids = token_type_ids.view(max_marker, bs, sent_len)


        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,  # list of (bs, sent_len, num_classes), len=n_head
            attention_mask=attention_mask,  # (bs, sent_len)
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,  # (B, topk, S')
            gate_topk_indices=gate_topk_indices,  # (B, topk, S')
            gate_all_scores=gate_all_scores,  # (B, L, S')
            loss=final_loss,
            input_ids=input_ids,  # (bs, sent_len) - already sent_mask filtered
            token_type_ids=token_type_ids,  # (max_marker, bs, sent_len)
            fusion_attentions=fusion_attentions,  # (B, S', n+1) or None
        )
