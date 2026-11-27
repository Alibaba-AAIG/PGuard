# Vendored PGuard EMA head (pguard_version=ema, seq variant), from
# Alibaba-AAIG/PGuard train/src/model/modeling_pguard_ema.py (Apache-2.0,
# inference path only) + the engine-facing score_step contract.
#
# Per-request multi-scale EMA state lives in a StatePool owned by the head:
#   prefill: chunks fold into the state in closed form (ema_fold, one
#            weighted-sum kernel instead of a per-token Python loop) and only
#            the chunk's LAST token is scored (the API aggregates prompt-side
#            risk from the last prefill record);
#   decode:  the EMA recurrence runs on EVERY step; the score can be thinned
#            via score=False (official PGUARD_COPY_LOGITS_PROB cadence).
#
# In-module self-test: python3 -m pguard_vllm.heads.ema

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertLayer

from ..state_pool import StatePool
from .base import GuardHeadBase, GuardHeadRegistry, HeadStepResult
from .common import (AdapterBlock, BasicClassificationHead, config_from_dict,
                     classifiers_forward, fused_adapter_forward,
                     PGuardMultiTasksOutputForCls)


def ema_fold(alphas, feats, state):
    """Exact multi-scale causal EMA fold, closed form:

        ema_n = sum_t a*(1-a)^(n-1-t)*x_t + (1-a)^n * ema_0

    alphas: (n_scales, H) fp32; feats: (C, n, H) fp32; state: (C, n_scales, H)
    fp32. Replaces the per-token Python loop (n kernel-launch rounds) with one
    weighted-sum einsum; same math as the official torch fallback
    (pguard/guard/history_state.py prefill_update).
    """
    C, n, H = feats.shape
    if n == 0:
        return state
    ret = 1.0 - alphas  # (n_scales, H)
    if n == 1:
        return (alphas.unsqueeze(0) * feats[:, 0, :].unsqueeze(1)
                + ret.unsqueeze(0) * state)
    # weight of token j (0-based): a * (1-a)^(n-1-j)
    k = torch.arange(n - 1, -1, -1, device=feats.device, dtype=torch.float32)
    w = alphas.unsqueeze(-1) * torch.exp(torch.log(ret).unsqueeze(-1) * k)
    within = torch.einsum("shn,cnh->csh", w, feats)  # (C, n_scales, H)
    decay = torch.exp(torch.log(ret) * float(n))     # (n_scales, H)
    return within + decay.unsqueeze(0) * state


class FrozenCausalEMA(nn.Module):
    """Fixed multi-scale EMA parameterized by effective decay lengths."""

    def __init__(self, hidden_size, effective_lengths=(25, 150)):
        super().__init__()
        self.n_heads = len(effective_lengths)
        self.hidden_size = hidden_size
        # alpha = 1 - exp(-1/tau), tau being the effective length
        init_alphas = [1.0 - math.exp(-1.0 / tau) for tau in effective_lengths]
        # Keep alphas as a plain attribute: DDP broadcasts registered buffers
        # at every forward, which can deadlock if another rank skips that
        # forward. These fixed values are identical on every rank and need no
        # synchronization.
        self.alphas = torch.tensor([
            [a] * hidden_size for a in init_alphas
        ])  # (n_heads, hidden_size)

    def forward(self, x, padding_mask=None, sent_mask=None):
        # x: (B, S, H), padding_mask: (B, S), sent_mask: (S,) boolean
        # sent_mask: when provided, only EMA outputs at sent_mask[i]=True
        # positions are stored, avoiding a full (B, S, H) tensor allocation
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

    Scales are sorted from longest to shortest, so information flows toward
    x_t; the x_t position provides the fused output.
    """

    def __init__(self, hidden_size, effective_lengths=(25, 150),
                 n_bert_layers=2, n_attn_heads=8):
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


@GuardHeadRegistry.register("ema")
class PluginGuardEMA(GuardHeadBase):
    """Multi-scale EMA head over the captured backbone layers."""

    def __init__(self, config):
        super().__init__(config)

        # EMA Attention Fusion
        self.ema_fusion = AttentionEMAFusionLayer(
            config.hidden_size, effective_lengths=[200, 150, 100, 50],
            n_bert_layers=1)

        # gating (gate-first: topk layer selection + soft aggregation; scoring
        # input passes a dedicated gate_norm first)
        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        bert_layer_config = BertConfig()
        bert_layer_config.add_cross_attention = False

        self.input_proj = nn.Linear(config.hidden_size,
                                    bert_layer_config.hidden_size)
        # FFN only, no attention needed
        self.adapter = nn.ModuleList(
            [AdapterBlock(bert_layer_config) for _ in range(1)])

        self.norm = nn.LayerNorm(config.hidden_size,
                                 eps=bert_layer_config.layer_norm_eps)
        # dedicated to gate scoring: one affine must not serve two distributions
        self.gate_norm = nn.LayerNorm(config.hidden_size,
                                      eps=bert_layer_config.layer_norm_eps)

        # build multi classification heads
        self.classifiers = nn.ModuleList()
        for name, num_classes in config.classifier["classification_heads"]:
            classifier_config = config_from_dict({
                "hidden_size": bert_layer_config.hidden_size,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes": num_classes,
                "dropout_type": config.classifier["dropout_type"],
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

    # -- vendored offline forward (training-faithful; prefill reuse) --------
    def forward_gate_first(self, features):
        """Score features (bs, layer, sent_len, hidden_size) before fusion.

        Returns topk_weights/topk_scores/topk_indices (B, topk, S) and
        all_scores (B, L, S); features and EMA reuse the same top-k weights.
        """
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

    def forward(self, features, token_type_ids=None, label_type=None,
                label_marker=None, output_attentions=False,
                attention_mask=None, **kwargs):
        """Offline full-sequence forward (kept for fidelity / parity checks);
        the engine drives the streaming score_step path instead."""
        # [1] EMA (computed over all captured layers)
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
            sent_mask = torch.stack(
                [mask for mask in token_type_ids.values()]).sum(0).sum(0) > 0  # (S,)
            # with no marker field of view at all (usually inference-side
            # truncation), sent_mask is all False: skip the filtering and
            # keep the full sequence
            if not sent_mask.any():
                sent_mask = None

        # [1a] EMA computed on the full sequence + sent_mask filtering (merged
        # into one step to avoid allocating the full EMA tensor)
        features_flat = features.reshape(B * L, S, H)
        ema_heads = self.ema_fusion.compute_ema(
            features_flat, padding_mask=ema_padding_mask, sent_mask=sent_mask)  # list of (B*L, S', H)

        # [1b] sent_mask filtering (features, token_type_ids, input_ids)
        input_ids = kwargs.get("input_ids")  # (bs, FULL_sent_len) or None
        if sent_mask is not None:
            features = features[:, :, sent_mask, :]  # (B, L, S', H)
            token_type_ids = {id: mask[:, sent_mask]
                              for id, mask in token_type_ids.items()}
            if input_ids is not None:
                input_ids = input_ids[:, sent_mask]

        # [2] gate-first: topk layer selection before fusion
        # NOTE: the gate consumes gate_norm-ed features; norm of the data path
        # happens after fusion
        S_prime = features.shape[2]
        gate_topk_weights, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate_first(features)  # (B, topk, S'), (B, topk, S'), (B, topk, S'), (B, L, S')
        n_scales = self.ema_fusion.n_ema_heads
        # merge the layer dim of features and EMA with the same set of topk weights
        expanded_indices = gate_topk_indices.unsqueeze(-1).expand(
            B, self.top_L, S_prime, H)
        topk_features = torch.gather(features, dim=1, index=expanded_indices)  # (B, topk, S', H)
        merged_cur = (topk_features * gate_topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, S', H)
        # ema_heads: list of (B*L, S', H) -> (B, L, n_scales, S', H)
        ema_stacked = torch.stack(ema_heads, dim=1).reshape(B, L, n_scales, S_prime, H)
        ema_expanded_indices = gate_topk_indices.unsqueeze(2).unsqueeze(-1).expand(
            B, self.top_L, n_scales, S_prime, H)
        topk_ema = torch.gather(ema_stacked, dim=1, index=ema_expanded_indices)  # (B, topk, n_scales, S', H)
        merged_ema = (topk_ema * gate_topk_weights.unsqueeze(2).unsqueeze(-1)).sum(dim=1)  # (B, n_scales, S', H)

        # [3] single BERT attention fusion (M=B*S' instead of B*L*S')
        merged_ema_heads = [merged_ema[:, i] for i in range(n_scales)]  # list of (B, S', H)
        fused_result = self.ema_fusion.fuse(
            merged_ema_heads, merged_cur, output_attentions=output_attentions)
        fusion_attentions = None
        if output_attentions:
            features, raw_attentions = fused_result  # (B, S', H)
            if len(raw_attentions) > 0:
                last_attn = raw_attentions[-1]  # (B*S', n_heads, n+1, n+1)
                n_players = self.ema_fusion.seq_len
                fusion_attentions = last_attn[:, :, -1, :].mean(dim=1)  # (B*S', n+1)
                fusion_attentions = fusion_attentions.reshape(B, S_prime, n_players)  # (B, S', n+1)
            # under sdpa raw_attentions is empty and fusion_attentions stays None
        else:
            features = fused_result  # (B, S', H)

        # [4] norm -> FFN -> cls
        features = self.norm(features)
        features = self.input_proj(features)  # (bs, sent_len, inter_hidden_size)
        if not self.training and self.fuse_ops:
            for layer in self.adapter:
                features = fused_adapter_forward(layer, features)
            full_logits_list = classifiers_forward(self.classifiers, features)
        else:
            for layer in self.adapter:
                features = layer(features)  # (bs, sent_len, inter_hidden_size)

            full_logits_list = []
            for classifier in self.classifiers:
                logits = classifier(features)  # (bs, sent_len, num_classes)
                full_logits_list.append(logits)

        attention_mask_out = None  # marker-derived mask; inference passes none

        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,
            attention_mask=attention_mask_out,
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,
            gate_topk_indices=gate_topk_indices,
            gate_all_scores=gate_all_scores,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            fusion_attentions=fusion_attentions,
        )

    # -- engine-facing scoring (GuardHeadBase contract) --------------------
    def _build_pool(self, device, max_slots):
        n_scales = self.ema_fusion.ema_module.alphas.shape[0]
        ema_shape = (len(self._layer_pos), n_scales,
                     next(self.parameters()).shape[-1])
        return StatePool(self.kind, device, ema_shape=ema_shape,
                         max_slots=max_slots, total_cls=self._total_cls)

    def _alphas(self):
        """(n_scales, H) fp32 alphas on the head's device, cached."""
        cached = getattr(self, "_pguard_alphas", None)
        device = next(self.parameters()).device
        if cached is not None and cached.device == device:
            return cached
        alphas = self.ema_fusion.ema_module.alphas.to(device=device,
                                                      dtype=torch.float32)
        self._pguard_alphas = alphas
        return alphas

    def score_step(self, prefill_rows, decode_rows,
                   score: bool = True) -> HeadStepResult | None:
        req_ids, flats, ns, decs = [], [], [], []
        if prefill_rows:
            r, f, n, d = self._score_prefill(prefill_rows)
            req_ids += r
            flats += f
            ns += n
            decs += d
        if decode_rows:
            out = self._score_decode(decode_rows, score=score)
            if out is not None:
                r, f, n, d = out
                req_ids += r
                flats += f
                ns += n
                decs += d
        if not req_ids:
            return None
        return HeadStepResult(req_ids, flats, ns, decs)

    def _score_prefill(self, rows):
        """Fold each prefill chunk into the request's EMA state (closed-form
        causal recurrence); score the chunk's LAST token only."""
        alphas = self._alphas()
        slot_t = self._pool.slots_for([rid for rid, _ in rows],
                                      rows[0][1].device)
        states_b = self._pool.ema_read(slot_t)  # (B, C, n_scales, H) fp32, zeros for new
        new_states, last_feats = [], []
        for i, (req_id, feats_all) in enumerate(rows):
            feats = feats_all[self._layer_pos]  # (C, n, H) fp32
            state = ema_fold(alphas, feats, states_b[i])
            new_states.append(state)
            last_feats.append(feats[:, -1, :])
        states_b = torch.stack(new_states, dim=0)             # (B, C, n_scales, H)
        self._pool.ema_write(slot_t, states_b)
        feats_b = torch.stack(last_feats, dim=0).unsqueeze(2)  # (B, C, 1, H)
        logits = self._score_batch(states_b, feats_b)
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1) for l in logits]
        req_ids = [rid for rid, _ in rows]
        cpu = self._to_cpu_flats(
            [[p[i].unsqueeze(0) for p in probs_all] for i in range(len(rows))],
            req_ids, [False] * len(rows))
        return req_ids, cpu, [1] * len(rows), [False] * len(rows)

    def _score_decode(self, rows, score=True):
        """Batched single-token decode: the EMA state update runs on EVERY
        step; scoring can be thinned via score=False (cadence)."""
        alphas = self._alphas()
        device = rows[0][1].device
        slot_t = self._pool.slots_for([rid for rid, _ in rows], device)
        batch_states = self._pool.ema_read(slot_t)  # (B, C, n_scales, H) fp32

        feats_b = torch.stack([f[self._layer_pos] for _, f in rows],
                              dim=0)  # (B, C, 1, H)
        xi = feats_b.squeeze(2)  # (B, C, H)
        batch_states = alphas.unsqueeze(0).unsqueeze(0) * xi.unsqueeze(2) \
            + (1.0 - alphas.unsqueeze(0).unsqueeze(0)) * batch_states

        self._pool.ema_write(slot_t, batch_states)
        if not score:
            return None
        B = len(rows)
        logits = self._score_batch(batch_states, feats_b)
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1) for l in logits]
        req_ids = [rid for rid, _ in rows]
        cpu = self._to_cpu_flats(
            [[p[i].unsqueeze(0) for p in probs_all] for i in range(B)],
            req_ids, [True] * B)
        return req_ids, cpu, [1] * B, [True] * B

    def _score_batch(self, batch_states, feats_b):
        """Gate -> merge -> fuse -> classify for a batch of tokens.

        batch_states: (B, C, n_scales, H); feats_b: (B, C, 1, H). Returns
        per-task logits of shape (B, 1, n_cls)."""
        dtype = self._dtype
        feats = feats_b.to(dtype)
        B, C, _, H = feats.shape
        scores = self.gate(self.gate_norm(feats)).squeeze(-1)  # (B, C, 1)
        if self.top_L >= C:
            # all captured layers are kept: the top-k gather is a no-op, the
            # weighted sum over the softmax weights is identical
            w = torch.softmax(scores, dim=1).unsqueeze(-1)     # (B, C, 1, 1)
            merged_cur = (feats * w).sum(dim=1)                # (B, 1, H)
            st = batch_states.to(dtype)
            merged_ema = (st * w).sum(dim=1)                   # (B, n_scales, H)
        else:
            topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)
            w = torch.softmax(topk_scores, dim=1)  # (B, topk, 1)
            merged_cur = (feats.gather(1, topk_indices.unsqueeze(-1)
                                       .expand(-1, -1, 1, H))
                          * w.unsqueeze(-1)).sum(dim=1)  # (B, 1, H)
            st = batch_states.to(dtype)
            st_sel = st.gather(1, topk_indices.unsqueeze(-1)
                               .expand(-1, -1, st.shape[2], st.shape[3]))
            merged_ema = (st_sel * w.unsqueeze(2)).sum(dim=1)  # (B, n_scales, H)
        ema_heads = [merged_ema[:, i].unsqueeze(1)
                     for i in range(merged_ema.shape[1])]
        fused = self.ema_fusion.fuse(ema_heads, merged_cur)  # (B, 1, H)
        h = self.norm(fused)
        h = self.input_proj(h)
        for layer in self.adapter:
            h = layer(h)
        return [classifier(h) for classifier in self.classifiers]


if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.heads.ema
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=16,
        classifier={"classification_heads": [["t_a", 3], ["t_b", 2]],
                    "dropout_prob": 0.0, "dropout_type": "stable"},
    )
    head = PluginGuardEMA(cfg)
    head.eval()
    layout = [("t_a", 0, 3), ("t_b", 3, 5)]
    head.bind(device="cpu", dtype=torch.float32, layer_pos=[0, 1, 2],
              total_cls=5, task_layout=layout, max_slots=4)

    # closed-form fold == the per-token loop (official torch fallback)
    alphas = torch.tensor([[0.3] * 16, [0.7] * 16])
    feats = torch.randn(3, 4, 16)          # (C, n, H)
    state0 = torch.randn(3, 2, 16)
    folded = ema_fold(alphas, feats, state0)
    ref = state0.clone()
    for t in range(4):
        ref = (alphas.unsqueeze(0) * feats[:, t, :].unsqueeze(1)
               + (1 - alphas).unsqueeze(0) * ref)
    assert torch.allclose(folded, ref, atol=1e-5), "ema_fold mismatch"

    # engine step: ragged prefill + decode; one flat vector per scored token
    pre = [("r1", torch.randn(3, 3, 16)), ("r2", torch.randn(3, 1, 16))]
    dec = [("r1", torch.randn(3, 1, 16))]
    res = head.score_step(pre, dec, score=True)
    assert res.req_ids == ["r1", "r2", "r1"]
    assert res.ns == [1, 1, 1]           # prefill scores the chunk's last token
    assert res.is_decodes == [False, False, True]
    assert all(len(f) == 5 for f in res.flats)
    assert all(abs(sum(f[:3]) - 1.0) < 1e-4 for f in res.flats)

    # cadence: score=False still advances the recurrence, returns nothing
    assert head.score_step([], dec, score=False) is None
    res2 = head.score_step([], dec, score=True)
    assert res2 is not None and res2.req_ids == ["r1"] and res2.ns == [1]

    # finalize frees the slot
    assert head.contains("r1") and head.contains("r2")
    fin = head.finalize("r1")
    assert fin is not None and len(fin) == 5 and not head.contains("r1")
    print("ema head self-test OK (ema_fold, engine step, cadence, finalize)")
