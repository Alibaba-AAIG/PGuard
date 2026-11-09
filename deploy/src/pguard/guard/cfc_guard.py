from __future__ import annotations
import math
import logging
from typing import Optional, Sequence, Tuple
from numbers import Integral
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertIntermediate,
    BertOutput,
    BertLayer,
)
from pguard.config import (
    PGUARD_CFC_LOOP_UNPACK as _CFC_LOOP_UNPACK,
    PGUARD_CFC_ENABLE_FUSION as _CFC_ENABLE_FUSION,
    PGUARD_CFC_TRITON_GRU as _CFC_TRITON_GRU,
    PGUARD_CFC_FUSE_ADAPTER_MIN_ROWS as _CFC_FUSE_ADAPTER_MIN_ROWS,
)

logger = logging.getLogger(__name__)


def validate_cfc_prefill_host_metadata(
    seq_lens_cpu,
    prefix_lens_cpu,
    *,
    batch_size: int,
    total_tokens: int,
    feature_tokens: int | None = None,
) -> int:
    if seq_lens_cpu is None:
        raise RuntimeError(
            "[CFC_GUARD] extend forward is missing extend_seq_lens_cpu; host-asynchronous prefill requires CPU sequence lengths"
        )
    if prefix_lens_cpu is None:
        raise RuntimeError(
            "CFC guard does not support radix-prefix reuse because the recurrent prefix state is unavailable"
        )
    if not all((isinstance(length, Integral) for length in seq_lens_cpu)):
        raise ValueError("extend_seq_lens_cpu must contain host integers")
    if not all((isinstance(length, Integral) for length in prefix_lens_cpu)):
        raise ValueError("extend_prefix_lens_cpu must contain host integers")
    if not isinstance(total_tokens, Integral):
        raise ValueError("CFC prefill total_tokens must be a host integer")
    if feature_tokens is not None and (not isinstance(feature_tokens, Integral)):
        raise ValueError("CFC prefill feature_tokens must be a host integer")
    seq_lens_host = [int(length) for length in seq_lens_cpu]
    prefix_lens_host = [int(length) for length in prefix_lens_cpu]
    total_tokens_host = int(total_tokens)
    feature_tokens_host = (
        total_tokens_host if feature_tokens is None else int(feature_tokens)
    )
    if (
        not seq_lens_host
        or len(seq_lens_host) != batch_size
        or len(prefix_lens_host) != batch_size
        or any((length <= 0 for length in seq_lens_host))
        or (sum(seq_lens_host) != total_tokens_host)
        or (feature_tokens_host < total_tokens_host)
    ):
        raise ValueError(
            f"CFC prefill CPU metadata does not match packed features: batch={batch_size}, seq_lens={seq_lens_host}, prefix_lens={prefix_lens_host}, tokens={total_tokens_host}, feature_tokens={feature_tokens_host}"
        )
    if any(prefix_lens_host):
        raise RuntimeError(
            "CFC guard does not support radix-prefix reuse because the recurrent prefix state is unavailable"
        )
    return max(seq_lens_host)


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    if _CFC_ENABLE_FUSION:
        raise
    triton = None
    tl = None
    _HAS_TRITON = False
libdevice = None
if _HAS_TRITON:
    try:
        from triton.language.extra.cuda import libdevice
    except ImportError:
        if _CFC_ENABLE_FUSION:
            raise
_HAS_LIBDEVICE = libdevice is not None
if _HAS_TRITON:

    @triton.jit
    def _cfc_pre_kernel(
        xz_ptr,
        xr_ptr,
        hzr_ptr,
        hprev_ptr,
        z_ptr,
        rh_ptr,
        mem,
        total,
        sx_row,
        sh_row,
        BLOCK: tl.constexpr,
        PRESERVE_ROUNDING: tl.constexpr = False,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        b = offs // mem
        e = offs % mem
        xz = tl.load(xz_ptr + b * sx_row + e, mask=m, other=0.0).to(tl.float32)
        xr = tl.load(xr_ptr + b * sx_row + e, mask=m, other=0.0).to(tl.float32)
        hz = tl.load(hzr_ptr + b * sh_row + e, mask=m, other=0.0).to(tl.float32)
        hr = tl.load(hzr_ptr + b * sh_row + mem + e, mask=m, other=0.0).to(tl.float32)
        hp = tl.load(hprev_ptr + b * mem + e, mask=m, other=0.0).to(tl.float32)
        if PRESERVE_ROUNDING:
            dtype = hprev_ptr.dtype.element_ty
            z = tl.sigmoid((xz + hz).to(dtype).to(tl.float32)).to(dtype)
            r = tl.sigmoid((xr + hr).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
        else:
            z = tl.sigmoid(xz + hz)
            r = tl.sigmoid(xr + hr)
        tl.store(z_ptr + b * mem + e, z.to(z_ptr.dtype.element_ty), mask=m)
        tl.store(rh_ptr + b * mem + e, (r * hp).to(rh_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _cfc_post_kernel(
        z_ptr,
        hh_ptr,
        xh_ptr,
        hprev_ptr,
        act_ptr,
        out_ptr,
        mem,
        total,
        sx_row,
        dt,
        BLOCK: tl.constexpr,
        PRESERVE_ROUNDING: tl.constexpr = False,
        USE_DT: tl.constexpr = False,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        b = offs // mem
        e = offs % mem
        z = tl.load(z_ptr + b * mem + e, mask=m, other=0.0).to(tl.float32)
        hh = tl.load(hh_ptr + b * mem + e, mask=m, other=0.0).to(tl.float32)
        xh = tl.load(xh_ptr + b * sx_row + e, mask=m, other=0.0).to(tl.float32)
        hp = tl.load(hprev_ptr + b * mem + e, mask=m, other=0.0).to(tl.float32)
        a = tl.load(act_ptr + b, mask=m, other=0.0).to(tl.float32)
        if PRESERVE_ROUNDING:
            dtype = hprev_ptr.dtype.element_ty
            candidate = (xh + hh).to(dtype).to(tl.float32)
            h_hat = libdevice.tanh(candidate).to(dtype).to(tl.float32)
            retained = (
                ((1.0 - z).to(dtype).to(tl.float32) * hp).to(dtype).to(tl.float32)
            )
            updated = (z * h_hat).to(dtype).to(tl.float32)
            h_new = (retained + updated).to(dtype).to(tl.float32)
            if USE_DT:
                delta = (h_new - hp).to(dtype).to(tl.float32)
                modulation = (dt * delta).to(dtype).to(tl.float32)
                h_new = (h_new + modulation).to(dtype).to(tl.float32)
            out = tl.where(a != 0, h_new, hp)
        else:
            h_hat = 2.0 * tl.sigmoid(2.0 * (xh + hh)) - 1.0
            h_new = hp + z * (h_hat - hp)
            h_new = h_new + dt * (h_new - hp)
            out = hp + a * (h_new - hp)
        tl.store(out_ptr + b * mem + e, out.to(out_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _gate_merge_kernel(
        feat_ptr,
        scores_ptr,
        out_ptr,
        H,
        temp,
        L: tl.constexpr,
        L_POW: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bpr = tl.cdiv(H, BLOCK)
        row = pid // bpr
        hcol = pid % bpr * BLOCK + tl.arange(0, BLOCK)
        mh = hcol < H
        l_offs = tl.arange(0, L_POW)
        lm = l_offs < L
        s = (
            tl.load(scores_ptr + row * L + l_offs, mask=lm, other=-float("inf")).to(
                tl.float32
            )
            / temp
        )
        w = tl.exp(s - tl.max(s, axis=0))
        w = w / tl.sum(w, axis=0)
        f = tl.load(
            feat_ptr + row * (L * H) + l_offs[:, None] * H + hcol[None, :],
            mask=lm[:, None] & mh[None, :],
            other=0.0,
        ).to(tl.float32)
        merged = tl.sum(w[:, None] * f, axis=0)
        tl.store(out_ptr + row * H + hcol, merged.to(out_ptr.dtype.element_ty), mask=mh)

    @triton.jit
    def _add_ln_kernel(
        x_ptr, res_ptr, w_ptr, b_ptr, out_ptr, N, eps, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        v = x + r
        mean = tl.sum(v, axis=0) / N
        d = tl.where(m, v - mean, 0.0)
        var = tl.sum(d * d, axis=0) / N
        y = d / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + row * N + offs, (y * w + b).to(out_ptr.dtype.element_ty), mask=m
        )


class CfcCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
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
            w_fp32 = torch.empty(weight.shape, dtype=torch.float32, device="cpu")
            torch.nn.init.orthogonal_(w_fp32)
            with torch.no_grad():
                weight.copy_(w_fp32.to(dtype=weight.dtype, device=weight.device))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: float) -> torch.Tensor:
        z = torch.sigmoid(
            torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz
        )
        r = torch.sigmoid(
            torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br
        )
        h_hat = torch.tanh(
            torch.matmul(x, self.Wh) + torch.matmul(r * h_prev, self.Uh) + self.bh
        )
        h_new = (1 - z) * h_prev + z * h_hat
        return h_new + dt * (h_new - h_prev)


class _CfcAdapter(nn.Module):
    def __init__(
        self, input_dim: int, mem_dim: int, use_dt: bool = False, dt: float = 1.0
    ):
        super().__init__()
        self.cfc = CfcCell(input_dim, mem_dim)
        self.use_dt = use_dt
        self.dt = float(dt)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        dt = self.dt if self.use_dt else 0.0
        return self.cfc(x_t, h, dt)


class _AdapterBlock(nn.Module):
    def __init__(self, config: BertConfig, use_triton: bool = False) -> None:
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _CFC_ENABLE_FUSION
            and self.use_triton
            and _HAS_TRITON
            and x.is_cuda
            and (not self.training)
            and (x.dim() >= 2)
            and x.is_contiguous()
            and (x.numel() // x.shape[-1] >= _CFC_FUSE_ADAPTER_MIN_ROWS)
        ):
            return self._forward_fused(x)
        return self.output(self.intermediate(x), x)

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        h = self.intermediate(x)
        d = self.output.dense(h)
        ln = self.output.LayerNorm
        N = ln.weight.numel()
        M = d.numel() // N
        out = torch.empty_like(d)
        _add_ln_kernel[M,](
            d, x, ln.weight, ln.bias, out, N, ln.eps, BLOCK=triton.next_power_of_2(N)
        )
        return out


class _BasicClassificationHead(nn.Module):
    def __init__(
        self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


class PrefillPool(nn.Module):
    def __init__(self, input_dim: int, mem_dim: int) -> None:
        super().__init__()
        self.k_proj = nn.Linear(input_dim, mem_dim)
        self.v_proj = nn.Linear(input_dim, mem_dim)
        self.query_vec = nn.Parameter(torch.randn(mem_dim) / math.sqrt(mem_dim))
        self.scale = math.sqrt(mem_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (k * self.query_vec).sum(-1) / self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -10000.0)
        w = torch.softmax(scores, dim=-1)
        return (w.unsqueeze(-1) * v).sum(1)


class FrozenCausalEMA(nn.Module):
    def __init__(self, hidden_size: int, effective_lengths: list = None) -> None:
        super().__init__()
        if effective_lengths is None:
            effective_lengths = [25, 150]
        self.n_heads = len(effective_lengths)
        self.hidden_size = hidden_size
        init_alphas = [1.0 - math.exp(-1.0 / tau) for tau in effective_lengths]
        self.alphas = torch.tensor([[a] * hidden_size for a in init_alphas])

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None, ema_state: list = None
    ):
        B, S, H = x.shape
        all_ema, finals = ([], [])
        for head_idx in range(self.n_heads):
            alpha = self.alphas[head_idx].to(device=x.device, dtype=x.dtype)
            if ema_state is None:
                ema = torch.zeros(B, H, device=x.device, dtype=x.dtype)
            else:
                ema = ema_state[head_idx].to(device=x.device, dtype=x.dtype)
            outputs = []
            for i in range(S):
                xi = x[:, i, :]
                if padding_mask is not None:
                    xi = xi * padding_mask[:, i].unsqueeze(-1).to(xi.dtype)
                ema = alpha * xi + (1 - alpha) * ema
                outputs.append(ema)
            all_ema.append(torch.stack(outputs, dim=1))
            finals.append(ema)
        return (all_ema, finals)


class PrefillEMASummarizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mem_dim: int,
        effective_lengths: list = None,
        n_bert_layers: int = 2,
        n_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        if effective_lengths is None:
            effective_lengths = [200, 150, 100, 50]
        effective_lengths = sorted(effective_lengths, reverse=True)
        self.ema = FrozenCausalEMA(input_dim, effective_lengths)
        self.n_ema_heads = self.ema.n_heads
        self.seq_len = self.n_ema_heads + 1
        self.input_dim = input_dim
        bert_config = BertConfig()
        bert_config.hidden_size = input_dim
        bert_config.intermediate_size = input_dim * 4
        bert_config.add_cross_attention = False
        bert_config.is_decoder = False
        bert_config.num_attention_heads = n_attn_heads
        self.bert_layers = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(n_bert_layers)]
        )
        self.out_proj = (
            nn.Identity() if input_dim == mem_dim else nn.Linear(input_dim, mem_dim)
        )

    def run_ema(
        self, x: torch.Tensor, ema_state: list = None, padding_mask: torch.Tensor = None
    ):
        return self.ema(x, padding_mask=padding_mask, ema_state=ema_state)

    def fuse(self, ema_vecs: list, x_boundary: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(ema_vecs + [x_boundary], dim=1)
        attn_mask = (
            torch.triu(
                torch.full(
                    (self.seq_len, self.seq_len),
                    torch.finfo(x_boundary.dtype).min,
                    device=x_boundary.device,
                    dtype=x_boundary.dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        for bert_layer in self.bert_layers:
            stacked = bert_layer(stacked, attention_mask=attn_mask)
        return self.out_proj(stacked[:, -1, :])


class CfcPluginGuard(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_cap: int,
        proj_dim: int = 768,
        mem_dim: int = 768,
        use_dt: bool = False,
        dt: float = 1.0,
        num_classes: int = 4,
        top_L: int = 3,
        gate_topk: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        graph_unroll: bool = False,
        use_triton: bool = True,
        prefill_mode: str = "pool",
        prefill_effective_lengths: tuple = (200, 150, 100, 50),
        classification_heads: Optional[Sequence[Tuple[str, int]]] = None,
    ) -> None:
        super().__init__()
        assert n_cap > 0, f"n_cap must be >0, got {n_cap}"
        self.hidden = hidden_size
        self.n_cap = n_cap
        self.proj_dim = proj_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.top_L = min(top_L, n_cap)
        self.gate_topk = gate_topk
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.gate = nn.Linear(hidden_size, 1)
        self.gate_temperature = 1.0
        self.input_proj = nn.Linear(hidden_size, proj_dim)
        self.adapter = _CfcAdapter(proj_dim, mem_dim, use_dt=use_dt, dt=dt)
        if proj_dim == mem_dim:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Linear(proj_dim, mem_dim, bias=False)
        adapter_cfg = BertConfig()
        adapter_cfg.hidden_size = mem_dim
        adapter_cfg.intermediate_size = mem_dim * 4
        self.post_adapter = _AdapterBlock(adapter_cfg, use_triton=use_triton)
        self.classification_heads = (
            [(str(name), int(n)) for name, n in classification_heads]
            if classification_heads
            else None
        )
        self.classifiers = nn.ModuleList()
        if self.classification_heads:
            self.num_classes = sum(n for _, n in self.classification_heads)
            for name, n_cls in self.classification_heads:
                self.classifiers.add_module(
                    name, _BasicClassificationHead(mem_dim, n_cls)
                )
        else:
            self.classifiers.append(_BasicClassificationHead(mem_dim, num_classes))
        self.prefill_mode = prefill_mode
        if prefill_mode == "ema":
            self.prefill_summarizer = PrefillEMASummarizer(
                proj_dim, mem_dim, effective_lengths=list(prefill_effective_lengths)
            )
            self.prefill_pool = None
        elif prefill_mode == "pool":
            self.prefill_summarizer = None
            self.prefill_pool = PrefillPool(proj_dim, mem_dim)
        else:
            raise ValueError(f"unknown prefill_mode: {prefill_mode}")
        self._Wx_fused = None
        self._bx_fused = None
        self._Uzr_fused = None
        self.graph_unroll = graph_unroll
        self.use_triton = use_triton
        self._unroll_graphs = {}
        self._graph_pool = None
        self._triton_active_cache = {}
        if use_triton and (not _HAS_TRITON):
            logger.warning("[CFC_GUARD] triton unavailable; using eager paths")
        self.to(dtype=dtype)
        self.eval()

    def _invalidate_fused_caches(self) -> None:
        self._Wx_fused = None
        self._bx_fused = None
        self._Uzr_fused = None
        self._unroll_graphs.clear()
        self._graph_pool = None

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse=recurse)
        self._invalidate_fused_caches()
        return module

    def train(self, mode: bool = True):
        module = super().train(mode)
        if mode:
            self._invalidate_fused_caches()
        return module

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        finally:
            self._invalidate_fused_caches()

    def encode_features(self, feat: torch.Tensor) -> torch.Tensor:
        w_dtype = self.norm.weight.dtype
        feat = feat.to(w_dtype)
        feat = self.norm(feat)
        merged = self._gate(feat)
        return self.input_proj(merged)

    def _gate(self, features: torch.Tensor) -> torch.Tensor:
        scores = self.gate(features).squeeze(-1)
        if (
            not self.gate_topk
            and _CFC_ENABLE_FUSION
            and self.use_triton
            and _HAS_TRITON
            and features.is_cuda
            and (features.dim() >= 2)
            and features.is_contiguous()
        ):
            return self._gate_merge_triton(features, scores)
        if self.gate_temperature != 1.0:
            scores = scores / self.gate_temperature
        if not self.gate_topk:
            weights = torch.softmax(scores, dim=-1)
            return (features * weights.unsqueeze(-1)).sum(dim=-2)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=-1)
        topk_features = torch.gather(
            features,
            -2,
            topk_indices.unsqueeze(-1).expand(*topk_indices.shape, self.hidden),
        )
        weights = torch.softmax(topk_scores, dim=-1).unsqueeze(-1)
        return (topk_features * weights).sum(dim=-2)

    def _gate_merge_triton(
        self, features: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        L, H = (self.n_cap, self.hidden)
        M = features.numel() // (L * H)
        merged = torch.empty(M, H, device=features.device, dtype=features.dtype)
        if M:
            grid = (M * triton.cdiv(H, 1024),)
            _gate_merge_kernel[grid](
                features.reshape(M, L, H),
                scores.reshape(M, L),
                merged,
                H,
                float(self.gate_temperature),
                L=L,
                L_POW=triton.next_power_of_2(L),
                BLOCK=1024,
            )
        return merged.reshape(features.shape[:-2] + (H,))

    def _build_fused_weights(self) -> None:
        cfc = self.adapter.cfc
        self._Wx_fused = torch.cat([cfc.Wz, cfc.Wr, cfc.Wh], dim=1).contiguous()
        self._bx_fused = torch.cat([cfc.bz, cfc.br, cfc.bh], dim=0).contiguous()
        self._Uzr_fused = torch.cat([cfc.Uz, cfc.Ur], dim=1).contiguous()

    def _gru_step_from_xproj(
        self, x_proj: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        x_z, x_r, x_h = x_proj.split(self.mem_dim, dim=-1)
        h_z, h_r = torch.matmul(h_prev, self._Uzr_fused).split(self.mem_dim, dim=-1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_hat = torch.tanh(x_h + torch.matmul(r * h_prev, self.adapter.cfc.Uh))
        h_new = (1 - z) * h_prev + z * h_hat
        if not self.adapter.use_dt:
            return h_new
        return h_new + self.adapter.dt * (h_new - h_prev)

    def _gru_step_triton(
        self,
        x_proj: torch.Tensor,
        h_buf: torch.Tensor,
        active: torch.Tensor,
        z_buf: torch.Tensor,
        rh_buf: torch.Tensor,
        out: torch.Tensor = None,
    ) -> None:
        cfc = self.adapter.cfc
        B, mem = h_buf.shape
        xz = x_proj[:, :mem]
        xr = x_proj[:, mem : 2 * mem]
        xh = x_proj[:, 2 * mem :]
        h_zr = torch.matmul(h_buf, self._Uzr_fused)
        total = B * mem
        grid = (triton.cdiv(total, 1024),)
        _cfc_pre_kernel[grid](
            xz,
            xr,
            h_zr,
            h_buf,
            z_buf,
            rh_buf,
            mem,
            total,
            xz.stride(0),
            h_zr.stride(0),
            BLOCK=1024,
            PRESERVE_ROUNDING=_CFC_ENABLE_FUSION,
            enable_fp_fusion=not _CFC_ENABLE_FUSION,
        )
        hh = torch.matmul(rh_buf, cfc.Uh)
        out_ptr = h_buf if out is None else out
        _cfc_post_kernel[grid](
            z_buf,
            hh,
            xh,
            h_buf,
            active,
            out_ptr,
            mem,
            total,
            xh.stride(0),
            dt=self.adapter.dt if self.adapter.use_dt else 0.0,
            BLOCK=1024,
            PRESERVE_ROUNDING=_CFC_ENABLE_FUSION,
            USE_DT=self.adapter.use_dt,
            enable_fp_fusion=not _CFC_ENABLE_FUSION,
        )

    def _gru_step_triton_pure(
        self, x_proj: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        h_buf = h_prev.contiguous()
        out = torch.empty_like(h_buf)
        if h_buf.numel() == 0:
            return out
        z_buf = torch.empty_like(h_buf)
        rh_buf = torch.empty_like(h_buf)
        self._gru_step_triton(
            x_proj, h_buf, self._triton_active_ones(h_buf), z_buf, rh_buf, out=out
        )
        return out

    def _triton_active_ones(self, h_buf: torch.Tensor) -> torch.Tensor:
        key = (h_buf.size(0), h_buf.device, h_buf.dtype)
        buf = self._triton_active_cache.get(key)
        if buf is None:
            buf = torch.ones(h_buf.size(0), device=h_buf.device, dtype=h_buf.dtype)
            self._triton_active_cache[key] = buf
        return buf

    def gru_step(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        if self._Wx_fused is None:
            self._build_fused_weights()
        x_proj = torch.matmul(x_t, self._Wx_fused) + self._bx_fused
        if (
            _CFC_ENABLE_FUSION
            and _CFC_TRITON_GRU
            and self.use_triton
            and _HAS_TRITON
            and x_t.is_cuda
            and (x_t.dim() == 2)
        ):
            return self._gru_step_triton_pure(x_proj, h_prev)
        return self._gru_step_from_xproj(x_proj, h_prev)

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.classifiers) == 1:
            return torch.softmax(self.classifiers[0](features), dim=-1)
        return torch.cat([torch.softmax(head(features), dim=-1) for head in self.classifiers], dim=-1)

    def score(self, h: torch.Tensor, cfc_input: torch.Tensor) -> torch.Tensor:
        features = h + self.skip_proj(cfc_input)
        features = self.post_adapter(features)
        logits = self._classify(features)
        return logits.float()

    def query_predict(self, h0: torch.Tensor):
        query_feat = self.post_adapter(h0.unsqueeze(1)).squeeze(1)
        query_logits = self._classify(query_feat)
        return query_logits.float()

    @torch.no_grad()
    def forward_step(self, feat: torch.Tensor, h_prev: torch.Tensor):
        cfc_input = self.encode_features(feat)
        h_new = self.gru_step(cfc_input, h_prev)
        logits = self.score(h_new, cfc_input)
        return (logits, h_new)

    def _unpack_prefill_input(
        self,
        cfc_input: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        seq_lens_host=None,
    ):
        batch_size = seq_lens.size(0)
        total_tokens = cfc_input.size(0)
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not _CFC_LOOP_UNPACK and total_tokens == batch_size * max_seq_len:
            return (cfc_input.view(batch_size, max_seq_len, self.proj_dim), None)
        if _CFC_LOOP_UNPACK:
            if seq_lens_host is None:
                raise RuntimeError(
                    "PGUARD_CFC_LOOP_UNPACK=1 requires host sequence lengths"
                )
            if not all((isinstance(length, Integral) for length in seq_lens_host)):
                raise ValueError("host sequence lengths must contain host integers")
            lengths = tuple((int(length) for length in seq_lens_host))
            if (
                len(lengths) != batch_size
                or sum(lengths) != total_tokens
                or max(lengths, default=0) != max_seq_len
                or any((length <= 0 for length in lengths))
            ):
                raise ValueError(
                    "host sequence lengths do not match packed prefill input"
                )
            padding = cfc_input.new_zeros(max_seq_len, self.proj_dim)
            rows = []
            offset = 0
            for length in lengths:
                row = cfc_input[offset : offset + length]
                rows.append(torch.cat((row, padding[: max_seq_len - length]), dim=0))
                offset += length
            padded = torch.stack(rows, dim=0)
            mask = torch.arange(max_seq_len, device=cfc_input.device).unsqueeze(
                0
            ) < seq_lens.unsqueeze(1)
            return (padded, mask)
        prompt_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=cfc_input.device),
            seq_lens,
            output_size=total_tokens,
        )
        offsets = torch.cumsum(seq_lens, dim=0) - seq_lens
        local_pos = (
            torch.arange(total_tokens, device=cfc_input.device) - offsets[prompt_idx]
        )
        flat_idx = prompt_idx * max_seq_len + local_pos
        padded = cfc_input.new_zeros(batch_size * max_seq_len, self.proj_dim)
        padded.index_copy_(0, flat_idx, cfc_input)
        padded = padded.view(batch_size, max_seq_len, self.proj_dim)
        mask = torch.arange(max_seq_len, device=cfc_input.device).unsqueeze(
            0
        ) < seq_lens.unsqueeze(1)
        return (padded, mask)

    @torch.no_grad()
    def prefill_parallel(
        self,
        feat: torch.Tensor,
        seq_lens: torch.Tensor,
        h0_init: torch.Tensor = None,
        prefill_state=None,
        *,
        max_seq_len: int,
        seq_lens_host=None,
        compute_query_logits: bool = True,
    ):
        device = feat.device
        seq_lens = seq_lens.to(device=device, dtype=torch.long)
        B = seq_lens.size(0)
        cfc_input = self.encode_features(feat)
        padded, mask = self._unpack_prefill_input(
            cfc_input, seq_lens, max_seq_len, seq_lens_host=seq_lens_host
        )
        if self.prefill_mode == "ema":
            ema_seqs, ema_final = self.prefill_summarizer.run_ema(
                padded, ema_state=prefill_state, padding_mask=mask
            )
            pos = (seq_lens - 1).clamp(min=0)
            b_idx = torch.arange(B, device=device)
            ema_vecs = [seq[b_idx, pos] for seq in ema_seqs]
            h0 = self.prefill_summarizer.fuse(ema_vecs, padded[b_idx, pos])
            query_logits = self.query_predict(h0) if compute_query_logits else None
            return (h0, query_logits, ema_final)
        h0 = self.prefill_pool(padded, mask=mask)
        query_logits = self.query_predict(h0) if compute_query_logits else None
        return (h0, query_logits, None)

    @torch.no_grad()
    def prefill_unroll(
        self,
        feat: torch.Tensor,
        h0: torch.Tensor,
        seq_lens: torch.Tensor,
        score: str = "last",
        sample_idx: torch.Tensor = None,
    ):
        assert score in ("none", "last", "all"), f"unknown score mode {score!r}"
        device = feat.device
        T = feat.size(0)
        seq_lens = seq_lens.to(device=device, dtype=torch.long)
        B = seq_lens.size(0)
        cfc_input = self.encode_features(feat)
        if self._Wx_fused is None:
            self._build_fused_weights()
        x_proj_all = torch.matmul(cfc_input, self._Wx_fused) + self._bx_fused
        offsets = torch.cumsum(seq_lens, dim=0) - seq_lens
        max_len = int(seq_lens.max().item())
        h = h0.to(self.adapter.cfc.Uh.dtype)
        need_collect = score == "all"
        h_tok = h.new_empty(T, self.mem_dim) if need_collect else None
        if self.graph_unroll and feat.is_cuda:
            h_final = self._unroll_graph_loop(
                x_proj_all, h, seq_lens, offsets, max_len, T, h_tok
            )
        else:
            for t in range(max_len):
                tok_idx = offsets + t
                active = seq_lens > t
                x_t = x_proj_all[tok_idx.clamp(max=T - 1)]
                h_new = self._gru_step_from_xproj(x_t, h)
                h = torch.where(active.unsqueeze(-1), h_new, h)
                if need_collect:
                    h_tok[tok_idx[active]] = h[active]
            h_final = h
        if score == "none":
            return (h_final, None)
        if score == "last":
            last_indices = offsets + seq_lens - 1
            return (h_final, self.score(h_final, cfc_input[last_indices]))
        if sample_idx is not None:
            return (h_final, self.score(h_tok[sample_idx], cfc_input[sample_idx]))
        return (h_final, self.score(h_tok, cfc_input))

    _UNROLL_GRAPH_CHUNK = 128

    def _capture_unroll_graph(self, B: int, device, dtype, need_collect: bool):
        C = self._UNROLL_GRAPH_CHUNK
        B_pad = 1 << max(B - 1, 0).bit_length()
        x_chunk = torch.zeros(C, B_pad, 3 * self.mem_dim, device=device, dtype=dtype)
        h_buf = torch.zeros(B_pad, self.mem_dim, device=device, dtype=dtype)
        active_chunk = torch.zeros(C, B_pad, dtype=torch.bool, device=device)
        h_snap = (
            torch.zeros(C, B_pad, self.mem_dim, device=device, dtype=dtype)
            if need_collect
            else None
        )
        use_triton = self.use_triton and _HAS_TRITON and x_chunk.is_cuda
        z_buf = rh_buf = None
        if use_triton:
            z_buf = torch.zeros(B_pad, self.mem_dim, device=device, dtype=dtype)
            rh_buf = torch.zeros(B_pad, self.mem_dim, device=device, dtype=dtype)

        def _chunk():
            for i in range(C):
                if use_triton:
                    self._gru_step_triton(
                        x_chunk[i], h_buf, active_chunk[i], z_buf, rh_buf
                    )
                else:
                    h_new = self._gru_step_from_xproj(x_chunk[i], h_buf)
                    h_buf.copy_(
                        torch.where(active_chunk[i].unsqueeze(-1), h_new, h_buf)
                    )
                if h_snap is not None:
                    h_snap[i].copy_(h_buf)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _chunk()
        torch.cuda.current_stream().wait_stream(s)
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._graph_pool):
            _chunk()
        self._unroll_graphs[B_pad, need_collect] = (
            g,
            x_chunk,
            h_buf,
            active_chunk,
            h_snap,
        )
        return B_pad

    def _unroll_graph_loop(self, x_proj_all, h, seq_lens, offsets, max_len, T, h_tok):
        device, B = (x_proj_all.device, seq_lens.size(0))
        C = self._UNROLL_GRAPH_CHUNK
        L_pad = (max_len + C - 1) // C * C
        t_range = torch.arange(L_pad, device=device).unsqueeze(1)
        tok_idx_table = (offsets.unsqueeze(0) + t_range).clamp(max=T - 1)
        active_table = seq_lens.unsqueeze(0) > t_range
        need_collect = h_tok is not None
        B_pad = 1 << max(B - 1, 0).bit_length()
        key = (B_pad, need_collect)
        if key not in self._unroll_graphs:
            self._capture_unroll_graph(B, device, x_proj_all.dtype, need_collect)
        g, x_chunk, h_buf, active_chunk, h_snap = self._unroll_graphs[key]
        h_buf[:B].copy_(h)
        active_chunk[:, B:].fill_(False)
        for c in range(L_pad // C):
            s0 = c * C
            x_chunk[:, :B].copy_(x_proj_all[tok_idx_table[s0 : s0 + C]])
            active_chunk[:, :B].copy_(active_table[s0 : s0 + C])
            g.replay()
            if h_tok is not None:
                tok_c = tok_idx_table[s0 : s0 + C]
                act_c = active_table[s0 : s0 + C]
                h_tok[tok_c[act_c]] = h_snap[:, :B][act_c]
        return h_buf[:B].clone()


_CFC_EXPECTED_VERSION = "gru"


def validate_cfc_guard_config(
    state_path: str, hidden_size: int, context: str = "cfc checkpoint"
) -> dict:
    if not state_path:
        return {}
    from .guard_utils import (
        PRODUCTION_CLASSIFICATION_HEADS,
        PRODUCTION_LABELS,
    )
    from .seq_guard import (
        load_seq_guard_config,
        resolve_classification_heads,
        validate_task_schema,
    )

    config = load_seq_guard_config(state_path)
    if not config:
        raise ValueError(
            f"{context} requires an adjacent config.json; a bare state file cannot "
            "prove its pguard_version or hidden size"
        )
    version = str(config.get("pguard_version", ""))
    if version != _CFC_EXPECTED_VERSION:
        raise ValueError(
            f"{context} pguard_version mismatch: expected "
            f"{_CFC_EXPECTED_VERSION!r}, got {version!r}"
        )
    if config.get("pguard_capture_layers") is None:
        raise ValueError(f"{context} pguard_capture_layers is required")
    heads = resolve_classification_heads(config)
    if tuple(heads) != PRODUCTION_CLASSIFICATION_HEADS:
        raise ValueError(f"{context} must use the exact production nine-task/38-class schema")
    validate_task_schema(config, heads, context=context, expected_labels=PRODUCTION_LABELS)
    configured_hidden = config.get("hidden_size")
    if configured_hidden is not None and int(configured_hidden) != int(hidden_size):
        raise ValueError(
            f"{context} hidden_size={int(configured_hidden)} conflicts with the "
            f"backbone's {int(hidden_size)}"
        )
    return config


def resolve_cfc_classification_heads(state_path: str):
    if not state_path:
        return None
    from .seq_guard import load_seq_guard_config, resolve_classification_heads

    return resolve_classification_heads(load_seq_guard_config(state_path))


def load_cfc_guard_state(
    guard: CfcPluginGuard, path: str, strict: bool = False
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict, meta = payload["state_dict"], payload.get("meta", {})
    else:
        state_dict, meta = payload, {}
    target_dtype = next(guard.parameters()).dtype
    target_device = next(guard.parameters()).device
    migrated = {
        k: v.to(dtype=target_dtype, device=target_device) for k, v in state_dict.items()
    }
    result = guard.load_state_dict(migrated, strict=strict)
    if not strict:
        if result.missing_keys:
            logger.info(
                "load_cfc_guard_state: %d missing keys (new v10 modules): %s",
                len(result.missing_keys),
                result.missing_keys[:5],
            )
        if result.unexpected_keys:
            logger.warning(
                "load_cfc_guard_state: %d unexpected keys: %s",
                len(result.unexpected_keys),
                result.unexpected_keys[:5],
            )
    return meta
