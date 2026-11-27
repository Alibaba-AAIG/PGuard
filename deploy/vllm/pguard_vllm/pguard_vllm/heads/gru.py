# Vendored PGuard GRU head (pguard_version=gru, cfc variant), from
# Alibaba-AAIG/PGuard train/src/model/modeling_pguard_gru.py (Apache-2.0,
# inference path only) + the engine-facing score_step contract.
#
# Per-request response recurrence state (mem_dim vector) lives in a StatePool
# owned by the head:
#   prefill: parallel query summarization -> h0 (pool attention pooling, or a
#            fixed-size EMA summary with PGUARD_CFC_PREFILL_MODE=ema) plus a
#            query-level score;
#   decode:  one CfC/GRU step per token; the recurrence runs on EVERY step,
#            the score can be thinned via score=False (official cadence).
#
# In-module self-test: python3 -m pguard_vllm.heads.gru

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput, BertLayer

from ..pguard_config import get_config
from ..state_pool import StatePool
from .base import GuardHeadBase, GuardHeadRegistry, HeadStepResult, pad_rows
from .common import BasicClassificationHead, config_from_dict, \
    classifiers_forward, fused_adapter_forward


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

    def __init__(self, hidden_size, effective_lengths=(25, 150)):
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

    The fixed-size EMA state carries exact summaries across chunks. Shorter
    time constants reduce dilution of isolated risk evidence.
    """

    def __init__(self, input_dim, mem_dim, effective_lengths=(200, 150, 100, 50),
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


@GuardHeadRegistry.register("gru")
class PluginGuardGRU(GuardHeadBase):
    """Query-summary + CfC recurrent head over the captured backbone layers."""

    def __init__(self, config):
        super().__init__(config)

        # gating
        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        # sequence adapter: pure CfCCell(GRU), no attention
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

        # parallel prefill: query segment -> CFC initial state h0
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

        # one AdapterBlock after the CfC GRU
        adapter_config = BertConfig()
        adapter_config.hidden_size = mem_dim
        adapter_config.intermediate_size = mem_dim * 4
        self.post_adapter = AdapterBlockGru(adapter_config)

        # classification heads
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

    # -- streaming API (the engine drives these through score_step) --------
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

    def query_predict(self, h0):
        """Return query_feat (B, mem_dim) and per-head logits (B, num_classes)."""
        query_feat = self.post_adapter(h0.unsqueeze(1)).squeeze(1)  # (B, mem_dim)
        query_logits_list = self._apply_classifiers(query_feat)
        return query_feat, query_logits_list

    def _apply_classifiers(self, x):
        """Classification heads over shared-trunk features (fused at eval)."""
        if not self.training and self.fuse_ops:
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
        if not self.training and self.fuse_ops:
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

    # -- engine-facing scoring (GuardHeadBase contract) --------------------
    def _build_pool(self, device, max_slots):
        return StatePool(self.kind, device, mem_dim=self.mem_dim,
                         max_slots=max_slots, total_cls=self._total_cls)

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
        """Batched query summarization -> h0 state + query-level scores."""
        rows = [(rid, f[self._layer_pos]) for rid, f in rows]
        padded, mask = pad_rows(rows)
        feats = padded.to(dtype=self._dtype)
        with torch.no_grad():
            query_logits_list, h0, _ = self.forward_prefill(feats, mask=mask)
        probs_all = [torch.softmax(ql.float(), dim=-1)
                     for ql in query_logits_list]  # (B, n_cls)
        req_ids = [rid for rid, _ in rows]
        cpu = self._to_cpu_flats(
            [[p[i].unsqueeze(0) for p in probs_all] for i in range(len(rows))],
            req_ids, [False] * len(rows))
        slot_t = self._pool.slots_for(req_ids, rows[0][1].device)
        self._pool.gru_write(slot_t, h0)
        return req_ids, cpu, [1] * len(rows), [False] * len(rows)

    def _score_decode(self, rows, score=True):
        """Batched single-token CfC/GRU step with carried hidden state; the
        state recurrence runs every step, scoring can be thinned."""
        device = rows[0][1].device
        slot_t = self._pool.slots_for([rid for rid, _ in rows], device)
        h_prev = self._pool.gru_read(slot_t)  # (B, mem_dim) fp32
        feats = torch.stack([f[self._layer_pos] for _, f in rows],
                            dim=0).to(self._dtype)
        with torch.no_grad():
            logits_list, h_new = self.forward_step(feats, h_prev,
                                                   compute_scores=score)
        self._pool.gru_write(slot_t, h_new)
        if not score:
            return None
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1)
                     for l in logits_list]  # (B, n_cls)
        req_ids = [rid for rid, _ in rows]
        cpu = self._to_cpu_flats(
            [[p[i].unsqueeze(0) for p in probs_all] for i in range(len(rows))],
            req_ids, [True] * len(rows))
        return req_ids, cpu, [1] * len(rows), [True] * len(rows)


if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.heads.gru
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=16, proj_dim=8, mem_dim=8, use_dt=False, dt=1.0,
        prefill_mode="pool",
        classifier={"classification_heads": [["t_a", 3]],
                    "dropout_prob": 0.0, "dropout_type": "stable"},
    )
    head = PluginGuardGRU(cfg)
    head.eval()
    head.bind(device="cpu", dtype=torch.float32, layer_pos=[0, 1, 2],
              total_cls=3, task_layout=[("t_a", 0, 3)], max_slots=4)

    # prefill: ragged query segments -> one query-level score per request
    pre = [("r1", torch.randn(3, 4, 16)), ("r2", torch.randn(3, 2, 16))]
    res = head.score_step(pre, [], score=True)
    assert res.req_ids == ["r1", "r2"] and res.ns == [1, 1]
    assert res.is_decodes == [False, False]
    assert all(len(f) == 3 and abs(sum(f) - 1.0) < 1e-4 for f in res.flats)

    # decode: state-only step (cadence) then a scored step
    dec = [("r1", torch.randn(3, 1, 16)), ("r2", torch.randn(3, 1, 16))]
    assert head.score_step([], dec, score=False) is None
    res2 = head.score_step([], dec, score=True)
    assert res2.req_ids == ["r1", "r2"] and res2.is_decodes == [True, True]
    assert res2.ns == [1, 1] and all(len(f) == 3 for f in res2.flats)

    # state must actually carry: a second scored step changes the logits input
    res3 = head.score_step([], dec, score=True)
    assert res3.flats != res2.flats

    # finalize frees the slot
    assert head.contains("r1") and head.contains("r2")
    fin = head.finalize("r2")
    assert fin is not None and not head.contains("r2")
    print("gru head self-test OK (prefill, decode, cadence, carry, finalize)")
