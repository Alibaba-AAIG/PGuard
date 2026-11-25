# Vendored PGuard stateful safety heads (EMA / GRU), adapted from
# Alibaba-AAIG/PGuard train/src/model (Apache-2.0):
#   modeling_pguard_ema.py -> PluginGuardEMA  (pguard_version=ema, seq variant)
#   modeling_pguard_gru.py -> PluginGuardGRU  (pguard_version=gru, cfc variant)
# The offline full-sequence forward is preserved verbatim (used for prefill
# scoring); the external-state streaming API (forward_prefill/forward_step)
# drives incremental decode. Training-only loss branches are kept for
# fidelity but are unreachable at inference (label_type=None).

import math
import os

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput, BertLayer

from .head import BasicClassificationHead, config_from_dict
from .head import PGuardMultiTasksOutputForCls
from .head import classifiers_forward, fused_adapter_forward, _FUSE_OPS


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
        # (device, dtype) -> cached causal attention mask; fuse() runs every
        # decode step, rebuilding the mask there would be a per-step alloc.
        self._causal_masks = {}

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

        # 2. standard causal attention mask (lower triangular), cached per
        #    (device, dtype): information flow long-scale -> short-scale -> x_t
        #    format: (1, 1, seq_len, seq_len) float, 0=attend, -inf=masked
        mask_key = (features.device, features.dtype)
        attn_mask = self._causal_masks.get(mask_key)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((self.seq_len, self.seq_len),
                           torch.finfo(features.dtype).min,
                           device=features.device, dtype=features.dtype),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            self._causal_masks[mask_key] = attn_mask

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


class PluginGuardEMA(nn.Module):

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
        if not self.training and _FUSE_OPS:
            for layer in self.adapter:
                features = fused_adapter_forward(layer, features)
            full_logits_list = classifiers_forward(self.classifiers, features)
        else:
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







class CfcCell(nn.Module):
    """GRU-like cell with a learned retention path for persistent risk state.

    Returns h_new + dt * (h_new - h_prev), where
    h_new = (1 - z) * h_prev + z * h_hat.
    """

    def __init__(self, input_dim=128, hidden_dim=128):
        super(CfcCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # fp32 explicitly: vLLM switches the global default dtype to bf16
        # before load_weights, and orthogonal_ init (geqrf) has no bf16 CPU kernel
        self.Wz = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.float32))
        self.Uz = nn.Parameter(torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))
        self.bz = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        self.Wr = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.float32))
        self.Ur = nn.Parameter(torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))
        self.br = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        self.Wh = nn.Parameter(torch.randn(input_dim, hidden_dim, dtype=torch.float32))
        self.Uh = nn.Parameter(torch.randn(hidden_dim, hidden_dim, dtype=torch.float32))
        self.bh = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:
            torch.nn.init.orthogonal_(weight)

    def _fused_weights(self):
        """(Wx, bx, Uzr) cached: 3 input GEMMs -> 1, 2 hidden GEMMs -> 1
        (official cfc_guard._build_fused_weights pattern)."""
        key = tuple((p.data_ptr(), p._version)
                    for p in (self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh))
        cache = getattr(self, "_pguard_fused_cache", None)
        if cache is None or cache[0] != key:
            w_x = torch.cat([self.Wz, self.Wr, self.Wh], dim=1).detach()
            b_x = torch.cat([self.bz, self.br, self.bh]).detach()
            u_zr = torch.cat([self.Uz, self.Ur], dim=1).detach()
            cache = (key, w_x, b_x, u_zr)
            self._pguard_fused_cache = cache
        return cache[1], cache[2], cache[3]

    def forward(self, x, h_prev, dt):
        """
        x:      (B, input_dim)
        h_prev: (B, hidden_dim)
        dt:     scalar or (B,)
        """
        if self.training:
            z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)
            r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)
            h_hat = torch.tanh(torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh)
        else:
            # fused: one GEMM over [Wz;Wr;Wh], one over [Uz;Ur]
            hid = self.hidden_dim
            w_x, b_x, u_zr = self._fused_weights()
            xg = torch.addmm(b_x, x, w_x)          # (B, 3*hidden)
            hg = torch.mm(h_prev, u_zr)            # (B, 2*hidden)
            z = torch.sigmoid(xg[:, :hid] + hg[:, :hid])
            r = torch.sigmoid(xg[:, hid:2 * hid] + hg[:, hid:])
            h_hat = torch.tanh(xg[:, 2 * hid:] + torch.mm(r * h_prev, self.Uh))
        h_new = (1 - z) * h_prev + z * h_hat

        return h_new + dt * (h_new - h_prev)


class CfcAdapter(nn.Module):
    """Single-token CfcCell wrapper with internal or caller-managed state."""

    def __init__(self, input_dim, mem_dim, use_dt=False, dt=1.0):
        super().__init__()
        self.cfc = CfcCell(input_dim, mem_dim)
        self.use_dt = use_dt
        self.dt = float(dt)
        self._h = None

    def reset_state(self, batch_size=None, device=None, dtype=None):
        self._h = None
        if batch_size is not None:
            if device is None:
                device = next(self.parameters()).device
            if dtype is None:
                dtype = next(self.parameters()).dtype
            self._h = torch.zeros(batch_size, self.cfc.hidden_dim, device=device, dtype=dtype)

    def _ensure_state(self, x_t):
        if (self._h is None or self._h.size(0) != x_t.size(0)
                or self._h.device != x_t.device or self._h.dtype != x_t.dtype):
            self.reset_state(batch_size=x_t.size(0), device=x_t.device, dtype=x_t.dtype)

    def step(self, x_t, h=None):
        """Update one token.

        x_t: (B, input_dim); h: optional (B, mem_dim) previous state.
        With h=None, use and update self._h. With explicit h, leave self._h
        untouched so the caller owns state persistence.
        Returns h_t: (B, mem_dim).
        """
        dt = self.dt if self.use_dt else 0.0

        if h is not None:
            # explicit-state mode: the internal cache is untouched; the caller owns state persistence
            return self.cfc(x_t, h, dt)

        self._ensure_state(x_t)
        self._h = self.cfc(x_t, self._h, dt)

        return self._h

    def forward(self, x_t):
        return self.step(x_t)


class AdapterBlockGru(nn.Module):
    """Position-wise BERT FFN with residual connection and LayerNorm."""

    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)
        return layer_output


class PrefillPool(nn.Module):
    """Summarize query features into h0 with parallel attention pooling."""

    def __init__(self, input_dim, mem_dim):
        super().__init__()
        self.k_proj = nn.Linear(input_dim, mem_dim)
        self.v_proj = nn.Linear(input_dim, mem_dim)
        self.query_vec = nn.Parameter(torch.randn(mem_dim) / math.sqrt(mem_dim))
        self.scale = math.sqrt(mem_dim)

    def forward(self, x, mask=None):
        """
        x:    (B, T, input_dim) projected query-segment features
        mask: (B, T) bool, True=valid query token (masks the rest when query
              lengths differ within the batch)
        Returns:
          h0: (B, mem_dim)
        """
        k = self.k_proj(x)                                    # (B, T, mem_dim)
        v = self.v_proj(x)                                    # (B, T, mem_dim)
        scores = (k * self.query_vec).sum(-1) / self.scale    # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=-1)                     # (B, T)
        return (w.unsqueeze(-1) * v).sum(1)                   # (B, mem_dim)


class FrozenCausalEMAGru(nn.Module):
    """Fixed multi-scale causal EMA parameterized by effective decay lengths."""

    def __init__(self, hidden_size, effective_lengths=[25, 150]):
        super().__init__()
        self.n_heads = len(effective_lengths)
        self.hidden_size = hidden_size
        # alpha = 1 - exp(-1/tau), tau being the effective length (new-sample weight; larger tau = longer memory)
        init_alphas = [1.0 - math.exp(-1.0 / tau) for tau in effective_lengths]
        # Keep alphas as a plain attribute: DDP broadcasts registered buffers at
        # every forward, which can deadlock if another rank skips that forward.
        # These fixed values are identical on every rank and need no synchronization.
        self.alphas = torch.tensor([
            [a] * hidden_size for a in init_alphas
        ])  # (n_heads, hidden_size)

    def forward(self, x, padding_mask=None, ema_state=None):
        """
        x: (B, S, H)
        padding_mask: (B, S), invalid positions get zeroed input
        ema_state: list of (B, H), the per-scale EMA states at the end of the previous chunk; None = start from zero
        Returns:
            all_ema: list of (B, S, H), per-position EMA at each scale
            finals:  list of (B, H), per-scale EMA states at the sequence end (for the next chunk's carry-over)
        """
        B, S, H = x.shape
        all_ema = []
        finals = []
        for head_idx in range(self.n_heads):
            alpha = self.alphas[head_idx].to(x.device)  # (H,) a plain attribute does not follow model.to(); sync the device manually
            if ema_state is None:
                ema = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            else:
                ema = ema_state[head_idx].to(device=x.device, dtype=x.dtype)
            outputs = []
            for i in range(S):
                xi = x[:, i, :]  # (B, H)
                if padding_mask is not None:
                    xi = xi * padding_mask[:, i].unsqueeze(-1).to(xi.dtype)
                ema = alpha * xi + (1 - alpha) * ema
                outputs.append(ema)
            all_ema.append(torch.stack(outputs, dim=1))  # (B, S, H)
            finals.append(ema)
        return all_ema, finals


class PrefillEMASummarizer(nn.Module):
    """Summarize prefill with multi-scale EMA and causal attention fusion.

    The fixed-size EMA state carries exact summaries across chunks. Shorter time
    constants reduce dilution of isolated risk evidence.
    """

    def __init__(self, input_dim, mem_dim, effective_lengths=[200, 150, 100, 50],
                 n_bert_layers=2, n_attn_heads=8):
        super().__init__()
        # descending order so that for i<j, h_ema_i carries no less history than h_ema_j
        effective_lengths = sorted(effective_lengths, reverse=True)
        self.ema = FrozenCausalEMA(input_dim, effective_lengths)
        self.n_ema_heads = self.ema.n_heads
        self.seq_len = self.n_ema_heads + 1  # n EMA tokens + 1 boundary token x_t
        self.input_dim = input_dim

        bert_config = BertConfig()
        bert_config.hidden_size = input_dim
        bert_config.intermediate_size = input_dim * 4
        bert_config.add_cross_attention = False
        bert_config.is_decoder = False
        bert_config.num_attention_heads = n_attn_heads
        self.bert_layers = nn.ModuleList([
            BertLayer(bert_config) for _ in range(n_bert_layers)
        ])
        self._causal_masks = {}  # (device, dtype) -> causal mask, see fuse()

        if input_dim == mem_dim:
            self.out_proj = nn.Identity()
        else:
            self.out_proj = nn.Linear(input_dim, mem_dim)

    def run_ema(self, x, ema_state=None, padding_mask=None):
        """Advance EMA over x (B, T, input_dim).

        ema_state contains optional previous (B, input_dim) states per scale.
        Returns per-position sequences (B, T, input_dim) and final states
        (B, input_dim) for the next chunk.
        """
        return self.ema(x, padding_mask=padding_mask, ema_state=ema_state)

    def fuse(self, ema_vecs, x_boundary):
        """Fuse per-scale ema_vecs and x_boundary, both (B, input_dim).

        Returns h0 (B, mem_dim) from the final causal-attention position.
        """
        stacked = torch.stack(ema_vecs + [x_boundary], dim=1)  # (B, n+1, H)

        # causal attention mask (lower triangular): information flows long-scale -> short-scale -> x_t
        mask_key = (x_boundary.device, x_boundary.dtype)
        attn_mask = self._causal_masks.get(mask_key)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((self.seq_len, self.seq_len),
                           torch.finfo(x_boundary.dtype).min,
                           device=x_boundary.device, dtype=x_boundary.dtype),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)
            self._causal_masks[mask_key] = attn_mask

        for bert_layer in self.bert_layers:
            stacked = bert_layer(stacked, attention_mask=attn_mask)

        return self.out_proj(stacked[:, -1, :])  # (B, mem_dim)


class PluginGuardGRU(nn.Module):

    def __init__(self, config):
        super().__init__()

        ### gating ###
        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        ### sequence adapter: pure CfCCell(GRU), no attention ###
        proj_dim = getattr(config, 'proj_dim', 768)
        mem_dim = getattr(config, 'mem_dim', 768)
        use_dt = getattr(config, 'use_dt', False)
        dt = getattr(config, 'dt', 1.0)

        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # map backbone hidden_size to the CfC input dim
        self.input_proj = nn.Linear(config.hidden_size, proj_dim)

        # Skip connection: align input dim to GRU output dim for residual addition
        # the GRU learns only the incremental correction; the skip connection keeps
        # the discriminative information of the original features
        if proj_dim == mem_dim:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Linear(proj_dim, mem_dim, bias=False)

        # CfcAdapter handles only the single-token CfcCell state update
        self.adapter = CfcAdapter(
            input_dim=proj_dim,
            mem_dim=mem_dim,
            use_dt=use_dt,
            dt=dt,
        )

        ### parallel prefill: query segment -> CFC initial state h0 ###
        # pool reuses materialized query states and avoids serial recurrence;
        # ema keeps a fixed-size state for exact cross-chunk carry-over.
        self.prefill_mode = getattr(config, 'prefill_mode', 'pool')
        prefill_effective_lengths = getattr(config, 'prefill_effective_lengths', [200, 150, 100, 50])
        if self.prefill_mode == 'ema':
            self.prefill_summarizer = PrefillEMASummarizer(
                proj_dim, mem_dim, effective_lengths=prefill_effective_lengths)
            self.prefill_pool = None
        elif self.prefill_mode == 'pool':
            self.prefill_summarizer = None
            self.prefill_pool = PrefillPool(proj_dim, mem_dim)
        else:
            raise ValueError(f"unknown prefill_mode: {self.prefill_mode}")

        ### one AdapterBlock after the CfC GRU ###
        adapter_config = BertConfig()
        adapter_config.hidden_size = mem_dim
        adapter_config.intermediate_size = mem_dim * 4
        self.post_adapter = AdapterBlockGru(adapter_config)

        ### classification heads ###
        self.classifiers = nn.ModuleList()
        classification_heads = config.classifier["classification_heads"]
        for name, num_classes in classification_heads:
            classifier_config = config_from_dict({
                "hidden_size": mem_dim,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes": num_classes,
                "dropout_type": config.classifier["dropout_type"]
            })
            classifier = BasicClassificationHead(classifier_config)
            self.classifiers.add_module(name, classifier)

        # Initialize weights
        self.apply(self._init_weights)
        self.adapter.cfc.reset_parameters()

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

    def forward_gate(self, features):
        # feature: (bs, layer, sent_len, hidden_size)
        B, L, S, H = features.shape

        scores = self.gate(features).squeeze(-1)   # (B, L, S)
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

        # topk features
        expanded_indices = topk_indices.unsqueeze(-1).expand(B, self.top_L, S, H)
        topk_features = torch.gather(features, dim=1, index=expanded_indices)  # (B, topk, S, H)

        weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)  # (B, topk, S, 1)
        final_feature = (topk_features * weights).sum(dim=1)  # (B, S, H)

        all_scores = F.softmax(scores_no_noise, dim=1)

        return final_feature, topk_scores, topk_indices, all_scores

    # External-state streaming API: forward_prefill returns query logits and h0;
    # forward_step advances one response token while the caller owns the state.
    @property
    def mem_dim(self):
        return self.adapter.cfc.hidden_dim

    def init_state(self, batch_size, device=None, dtype=None):
        """Return an all-zero initial state (batch_size, mem_dim)."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.mem_dim, device=device, dtype=dtype)

    def encode_features(self, features):
        """Normalize and gate features (B, L, S, hidden_size).

        Returns cfc_input (B, S, proj_dim) and the gate score/index tensors.
        """
        features = self.norm(features)
        features, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate(features)  # (B, S, H)
        cfc_input = self.input_proj(features)  # (B, S, proj_dim)
        return cfc_input, gate_topk_scores, gate_topk_indices, gate_all_scores

    def query_predict(self, h0):
        """Return query_feat (B, mem_dim) and per-head logits (B, num_classes)."""
        query_feat = self.post_adapter(h0.unsqueeze(1)).squeeze(1)  # (B, mem_dim)
        query_logits_list = self._apply_classifiers(query_feat)
        return query_feat, query_logits_list

    def _apply_classifiers(self, x):
        """Classification heads over shared-trunk features (fused at eval)."""
        if not self.training and _FUSE_OPS:
            return classifiers_forward(self.classifiers, x)
        return [classifier(x) for classifier in self.classifiers]

    def forward_prefill(self, features, mask=None, prefill_state=None):
        """Summarize query features (B, L, T, hidden_size) in parallel.

        mask (B, T) marks valid query tokens. EMA mode accepts and returns
        per-scale (B, proj_dim) carry-over states; pool mode ignores them.
        Returns per-head query logits (B, num_classes), h0 (B, mem_dim), and
        the updated prefill state.
        """
        cfc_input, _, _, _ = self.encode_features(features)  # (B, T, proj_dim)
        B, T, P = cfc_input.shape

        if self.prefill_mode == 'ema':
            ema_seqs, ema_final = self.prefill_summarizer.run_ema(
                cfc_input, ema_state=prefill_state, padding_mask=mask)
            # query boundary = the last valid position of each sample (the EMA is
            # causal, so reading that position leaks no later tokens)
            if mask is None:
                pos = torch.full((B,), T - 1, dtype=torch.long, device=cfc_input.device)
            else:
                pos = (T - 1) - mask.flip(1).long().argmax(dim=1)
                pos = pos.clamp(min=0)
            b_idx = torch.arange(B, device=cfc_input.device)
            ema_vecs = [seq[b_idx, pos] for seq in ema_seqs]
            h0 = self.prefill_summarizer.fuse(ema_vecs, cfc_input[b_idx, pos])
            _, query_logits_list = self.query_predict(h0)
            return query_logits_list, h0, ema_final

        h0 = self.prefill_pool(cfc_input, mask=mask)  # (B, mem_dim)
        _, query_logits_list = self.query_predict(h0)
        return query_logits_list, h0, None

    def gru_unroll(self, cfc_input, init_state=None, active_mask=None):
        """Unroll cfc_input (B, S, proj_dim) without touching internal state.

        init_state (B, mem_dim) defaults to zeros. active_mask (B, S) freezes
        the state at False positions. Returns gru_output (B, S, mem_dim) and
        h_final (B, mem_dim).
        """
        B = cfc_input.size(0)
        if init_state is None:
            h = self.init_state(B, device=cfc_input.device, dtype=cfc_input.dtype)
        else:
            h = init_state.to(device=cfc_input.device, dtype=cfc_input.dtype)
        h_list = []
        for t in range(cfc_input.size(1)):
            h_prev = h
            h_new = self.adapter.step(cfc_input[:, t, :], h=h_prev)  # pure functional mode
            if active_mask is not None:
                h = torch.where(active_mask[:, t].unsqueeze(-1), h_new, h_prev)
            else:
                h = h_new
            h_list.append(h)
        gru_output = torch.stack(h_list, dim=1)  # (B, S, mem_dim)
        return gru_output, h

    def post_gru_forward(self, gru_output, cfc_input):
        """skip connection + AdapterBlock, shared by training and streaming."""
        # Skip connection: the GRU learns only the incremental correction, keeping
        # the discriminative information of the input_proj features
        features = gru_output + self.skip_proj(cfc_input)  # (B, S, mem_dim)
        if not self.training and _FUSE_OPS:
            return fused_adapter_forward(self.post_adapter, features)
        features = self.post_adapter(features)  # (B, S, mem_dim)
        return features

    def forward_step(self, features, h_prev, compute_scores=True):
        """Advance one token without touching internal state.

        features: (B, L, 1, hidden_size); h_prev: (B, mem_dim).
        compute_scores=False runs only the state recurrence (used on
        non-scoring decode steps); returns (None, h_new) then.
        Returns per-head logits (B, 1, num_classes) and h_new (B, mem_dim).
        The caller persists h_new for the next step.
        """
        cfc_input, _, _, _ = self.encode_features(features)  # (B, 1, proj_dim)
        h_prev = h_prev.to(device=cfc_input.device, dtype=cfc_input.dtype)
        h_new = self.adapter.step(cfc_input[:, 0, :], h=h_prev)  # (B, mem_dim)
        if not compute_scores:
            return None, h_new
        features_out = self.post_gru_forward(h_new.unsqueeze(1), cfc_input)
        full_logits_list = self._apply_classifiers(features_out)
        return full_logits_list, h_new

    def forward(
        self,
        features: torch.Tensor,  # (bs, layer, sent_len, hidden_size)
        token_type_ids: dict,    # dict(key=int, value=(bs, sent_len))
        label_type: str,
        label_marker: dict,
        assistant_start: Optional[Union[int, torch.Tensor]] = None,  # response start position (query/response boundary)
        init_state: Optional[torch.Tensor] = None,  # (B, mem_dim) initial state injected by an external manager; None=all zeros
        return_state: bool = False,  # when True, outputs final_state for the caller to write back
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], PGuardMultiTasksOutputForCls]:
        '''Parallel prefill + serial response; see the module docstring.'''

        # 0. norm + dynamic topk layer + input_proj
        cfc_input, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.encode_features(features)
        B, S, P = cfc_input.shape
        device = cfc_input.device

        # 1. assistant_start -> per-sample response start (B,)
        #    None/0 -> fully serial from the first token, empty query segment
        if assistant_start is None:
            start = torch.zeros(B, dtype=torch.long, device=device)
        elif isinstance(assistant_start, int):
            start = torch.full((B,), assistant_start, dtype=torch.long, device=device)
        else:
            start = assistant_start.to(device=device, dtype=torch.long)
            if start.dim() == 0:
                start = start.expand(B)
        start = start.clamp(min=0, max=S)

        pos_idx = torch.arange(S, device=device)
        in_query = pos_idx[None, :] < start[:, None]    # (B, S) query region
        is_response = ~in_query                         # (B, S) response region (start included)

        # 2. parallel prefill: query segment -> h0 (risk-state summary) + query-level prediction
        if init_state is None:
            base_h = self.init_state(B, device=device, dtype=cfc_input.dtype)
        else:
            base_h = init_state.to(device=device, dtype=cfc_input.dtype)
        if (start > 0).any():
            b_idx = torch.arange(B, device=device)
            boundary = (start - 1).clamp(min=0)
            if self.prefill_mode == 'ema':
                # causal EMA recursion: reading at the boundary position aggregates
                # only query-segment tokens, leaking no response tokens
                ema_seqs, _ = self.prefill_summarizer.run_ema(cfc_input)
                ema_vecs = [seq[b_idx, boundary] for seq in ema_seqs]
                h0_pool = self.prefill_summarizer.fuse(ema_vecs, cfc_input[b_idx, boundary])
            else:
                h0_pool = self.prefill_pool(cfc_input, mask=in_query)
            # samples with start=0 have no query segment: use zero/init_state
            h0 = torch.where((start > 0).unsqueeze(-1), h0_pool.to(base_h.dtype), base_h)
        else:
            h0 = base_h
        query_feat, query_logits_list = self.query_predict(h0)

        # 3. response region: serial per-token CFC starting from h0 (same math as the streaming forward_step)
        gru_output, h_final = self.gru_unroll(cfc_input, init_state=h0, active_mask=is_response)

        # 4. skip + AdapterBlock; the query region is filled with the h0 summary
        # features (the query-level prediction broadcast over query positions)
        features = self.post_gru_forward(gru_output, cfc_input)  # (B, S, mem_dim)
        features = torch.where(
            is_response.unsqueeze(-1),
            features,
            query_feat.unsqueeze(1).expand_as(features),
        )

        # 5. classifier (logits at every token; the query region carries the broadcast query-level prediction)
        full_logits_list = self._apply_classifiers(features)

        # preparation
        attention_mask = None
        if token_type_ids is not None:
            bs, sent_len = features.shape[:2]
            max_marker = len(token_type_ids)
            n_head = len(full_logits_list)
            marker_ids = list(token_type_ids.keys())

            token_type_ids = torch.stack([token_type_ids[k] for k in marker_ids], 0)  # (M, B, S)
            attention_mask = (token_type_ids.sum(0) > 0).to(torch.long)  # (B, S)

            # label source of the query-level supervision: the last marker inside
            # the query region (the "once a risk appears, all later slices are
            # risky" prior guarantees it equals the overall query risk)
            q_mask = (token_type_ids > 0) & in_query.unsqueeze(0)   # (M, B, S)
            q_mask_b = q_mask.permute(1, 0, 2).reshape(bs, -1)      # (B, M*S)
            q_any = q_mask_b.any(dim=1)                             # (B,)
            last_flat = q_mask_b.shape[1] - 1 - q_mask_b.flip(1).long().argmax(dim=1)
            query_marker_idx = last_flat // sent_len                # (B,)
            b_idx = torch.arange(bs, device=device)

            # exclude query-region markers from the per-token loss: h0 sees the
            # whole query, and the safe labels of slices before a risk appears
            # contradict that causally; query risk is supervised by the query-level loss alone
            token_type_ids = token_type_ids * is_response.unsqueeze(0).to(token_type_ids.dtype)

        # loss: per-token on response-region markers + the query-level loss
        final_loss = None
        if token_type_ids is not None and label_type is not None and label_marker is not None:
            if label_type != 'single':
                raise NotImplementedError(label_type)

            # stack markers into the bs dimension
            token_type_ids = token_type_ids.flatten(0, 1)  # (bs', S)

            logits_list = [x.unsqueeze(0).expand(max_marker, -1, -1, -1) for x in full_logits_list]
            logits_list = [x.flatten(0, 1) for x in logits_list]  # list of (bs', S, C)

            labels_mb = [torch.stack([label_marker[k]['batch_single_labels'][i] for k in marker_ids], 0)
                         for i in range(n_head)]  # list of (M, B)
            labels = [x.flatten(0, 1) for x in labels_mb]  # list of (bs',)
            labels = [x.unsqueeze(-1).expand(-1, sent_len) for x in labels]  # list of (bs', S)

            task_mask_mb = torch.stack([label_marker[k]['batch_task_mask'] for k in marker_ids], 0)  # (M, B, n_head)
            task_mask = task_mask_mb.flatten(0, 1)  # (bs', n_head)

            # ---- query-level supervision: h0's prediction vs the label of the last query-region marker ----
            query_losses = []
            for idx in range(n_head):
                q_label = labels_mb[idx][query_marker_idx, b_idx].to(device)  # (B,)
                q_loss = F.cross_entropy(query_logits_list[idx], q_label, reduction='none')  # (B,)
                query_losses.append(q_loss)
            query_losses = torch.stack(query_losses, 1)           # (B, n_head)
            query_task = task_mask_mb[query_marker_idx, b_idx].to(device)     # (B, n_head)
            query_valid = query_task.float() * q_any.unsqueeze(-1).float()    # (B, n_head)
            q_valid_counts = query_valid.sum()

            if (token_type_ids > 0).sum() == 0:
                print('No valid response-region tokens; response loss is skipped')
                if q_valid_counts > 0:
                    final_loss = (query_losses * query_valid).sum() / q_valid_counts.clamp(min=1e-8)
                else:
                    final_loss = torch.tensor(0.0, requires_grad=True, device=device)
            else:
                task_mask *= ((token_type_ids > 0).sum(1) > 0).float().unsqueeze(-1)

                multitask_loss = []
                for idx in range(n_head):
                    logits = logits_list[idx]   # (bs', S, C)
                    labels_ = labels[idx]       # (bs', S)

                    loss = nn.CrossEntropyLoss(reduction="none")(
                        logits.flatten(0, 1), labels_.flatten(0, 1)
                    ).view(-1, sent_len)  # (bs', S)

                    loss *= (token_type_ids > 0).float()
                    valid_token_counts = (token_type_ids > 0).float().sum(1).clamp(min=1e-8)
                    loss = loss.sum(1) / valid_token_counts

                    multitask_loss.append(loss)

                multitask_loss = torch.stack(multitask_loss, 1)  # (bs', n_head)
                multitask_loss *= task_mask

                # normalize the response per-token loss and the query-level loss
                # together: both are counted in (sample, head) pairs, the
                # denominator is the sum of valid pairs of both parts, keeping the
                # scales balanced
                valid_task_counts = task_mask.float().sum().clamp(min=1e-8)
                final_loss = (multitask_loss.sum() + (query_losses * query_valid).sum()) \
                    / (valid_task_counts + q_valid_counts)

            # restore shapes
            token_type_ids = token_type_ids.view(max_marker, bs, sent_len)

        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,
            attention_mask=attention_mask,
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,
            gate_topk_indices=gate_topk_indices,
            gate_all_scores=gate_all_scores,
            loss=final_loss,
            input_ids=kwargs.get('input_ids'),
            token_type_ids=token_type_ids,
            # with return_state=True the end-of-sequence hidden state is returned
            # for the external manager to write back (continuing the next chunk)
            final_state=h_final if return_state else None,
        )
