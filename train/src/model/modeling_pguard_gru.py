# Parallel query prefill + serial CfC/GRU response. The query summary h0 seeds
# the response state and produces a query-level prediction. External streaming
# callers can use forward_prefill() and forward_step() for the same two paths.
# Response markers receive per-token CE; query supervision uses only the last
# query marker because h0 represents the whole query. assistant_start=None/0
# selects a fully serial pass with zero initial state.

import math

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput, BertLayer

from src.utils.model_utils import BasicClassificationHead, config_from_dict
from src.utils.modeling_outputs import PGuardMultiTasksOutputForCls


class CfcCell(nn.Module):
    """GRU-like cell with a learned retention path for persistent risk state.

    Returns h_new + dt * (h_new - h_prev), where
    h_new = (1 - z) * h_prev + z * h_hat.
    """

    def __init__(self, input_dim=128, hidden_dim=128):
        super(CfcCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Wz = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uz = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bz = nn.Parameter(torch.zeros(hidden_dim))

        self.Wr = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Ur = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.br = nn.Parameter(torch.zeros(hidden_dim))

        self.Wh = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Uh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.Wz, self.Uz, self.Wr, self.Ur, self.Wh, self.Uh]:
            torch.nn.init.orthogonal_(weight)

    def forward(self, x, h_prev, dt):
        """
        x:      (B, input_dim)
        h_prev: (B, hidden_dim)
        dt:     scalar or (B,)
        """
        z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh)
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


class AdapterBlock(nn.Module):
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


class FrozenCausalEMA(nn.Module):
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
        attn_mask = torch.triu(
            torch.full((self.seq_len, self.seq_len), torch.finfo(x_boundary.dtype).min,
                       device=x_boundary.device, dtype=x_boundary.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, n+1, n+1)

        for bert_layer in self.bert_layers:
            stacked = bert_layer(stacked, attention_mask=attn_mask)

        return self.out_proj(stacked[:, -1, :])  # (B, mem_dim)


class PluginGuard(nn.Module):

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
        self.post_adapter = AdapterBlock(adapter_config)

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
        query_logits_list = [classifier(query_feat) for classifier in self.classifiers]
        return query_feat, query_logits_list

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
        features = self.post_adapter(features)  # (B, S, mem_dim)
        return features

    def forward_step(self, features, h_prev):
        """Advance one token without touching internal state.

        features: (B, L, 1, hidden_size); h_prev: (B, mem_dim).
        Returns per-head logits (B, 1, num_classes) and h_new (B, mem_dim).
        The caller persists h_new for the next step.
        """
        cfc_input, _, _, _ = self.encode_features(features)  # (B, 1, proj_dim)
        h_prev = h_prev.to(device=cfc_input.device, dtype=cfc_input.dtype)
        h_new = self.adapter.step(cfc_input[:, 0, :], h=h_prev)  # (B, mem_dim)
        features_out = self.post_gru_forward(h_new.unsqueeze(1), cfc_input)
        full_logits_list = [classifier(features_out) for classifier in self.classifiers]
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
        full_logits_list = []
        for classifier in self.classifiers:
            logits = classifier(features)  # (B, S, num_classes)
            full_logits_list.append(logits)

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
