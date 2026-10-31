from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def can_use_fused_sampled_ema(
    feat_seq: torch.Tensor,
    seq_lens: torch.Tensor,
    offsets: torch.Tensor,
    ema_old: torch.Tensor,
    prompt_idx: torch.Tensor,
    local_pos: torch.Tensor,
    sample_idx: torch.Tensor,
    alphas: torch.Tensor,
    chunk_size: int,
) -> bool:
    """Check only shape/device/layout metadata; never read CUDA tensor values."""
    if not (
        triton is not None
        and feat_seq.is_cuda
        and feat_seq.dim() == 2
        and feat_seq.size(0) > 0
        and feat_seq.size(1) > 0
        and feat_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and feat_seq.is_contiguous()
        and ema_old.dim() == 3
        and ema_old.device == feat_seq.device
        and ema_old.dtype == feat_seq.dtype
        and alphas.dim() == 1
        and alphas.numel() > 0
        and alphas.device == feat_seq.device
        and alphas.dtype == torch.float32
        and alphas.is_contiguous()
        and chunk_size in (48, 64)
    ):
        return False
    indices = (seq_lens, offsets, prompt_idx, local_pos, sample_idx)
    if not all(
        tensor.device == feat_seq.device
        and tensor.dim() == 1
        and tensor.dtype in (torch.int32, torch.int64)
        and tensor.is_contiguous()
        for tensor in indices
    ):
        return False
    return bool(
        seq_lens.numel() > 0
        and offsets.numel() == seq_lens.numel()
        and prompt_idx.numel() == feat_seq.size(0)
        and local_pos.numel() == feat_seq.size(0)
        and sample_idx.numel() > 0
        and ema_old.shape == (seq_lens.numel(), alphas.numel(), feat_seq.size(1))
    )


if triton is not None:
    @triton.jit
    def _chunk_end_kernel(
        feat,
        prompt_of_chunk,
        start_local,
        len_j,
        offsets,
        alphas,
        powers,
        out,
        D: tl.constexpr,
        T1: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        j = tl.program_id(0)
        scale = tl.program_id(1)
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

        length = tl.load(len_j + j)
        prompt = tl.load(prompt_of_chunk + j)
        start = tl.load(start_local + j)
        offset = tl.load(offsets + prompt)
        alpha = tl.load(alphas + scale)
        value = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for c in tl.static_range(0, BLOCK_C):
            valid = (c < C) & (c < length)
            x = tl.load(
                feat + (offset + start + c) * D + d,
                mask=valid & (d < D),
                other=0.0,
            )
            power = tl.load(
                powers + scale * T1 + tl.maximum(length - 1 - c, 0),
                mask=valid,
                other=0.0,
            )
            value += x.to(tl.float32) * (alpha * power)
        tl.store(out + (j * tl.num_programs(1) + scale) * D + d, value, mask=d < D)

    @triton.jit
    def _sampled_output_kernel(
        feat,
        sample_idx,
        prompt_idx,
        local_pos,
        chunk_id,
        start_local,
        offsets,
        carry,
        alphas,
        powers,
        out,
        D: tl.constexpr,
        T1: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        m = tl.program_id(0)
        scale = tl.program_id(1)
        d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

        token_idx = tl.load(sample_idx + m)
        prompt = tl.load(prompt_idx + token_idx)
        local = tl.load(local_pos + token_idx)
        chunk = tl.load(chunk_id + token_idx)
        start = tl.load(start_local + chunk)
        pos = local - start
        offset = tl.load(offsets + prompt)
        alpha = tl.load(alphas + scale)
        within = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for c in tl.static_range(0, BLOCK_C):
            valid = (c < C) & (c <= pos)
            x = tl.load(
                feat + (offset + start + c) * D + d,
                mask=valid & (d < D),
                other=0.0,
            )
            power = tl.load(
                powers + scale * T1 + tl.maximum(pos - c, 0),
                mask=valid,
                other=0.0,
            )
            within += x.to(tl.float32) * (alpha * power)
        carry_value = tl.load(
            carry + (chunk * tl.num_programs(1) + scale) * D + d,
            mask=d < D,
            other=0.0,
        )
        decay = tl.load(powers + scale * T1 + pos + 1)
        tl.store(
            out + (m * tl.num_programs(1) + scale) * D + d,
            within + decay * carry_value,
            mask=d < D,
        )


def sampled_chunk_end(
    feat_seq: torch.Tensor,
    prompt_of_chunk: torch.Tensor,
    start_local: torch.Tensor,
    len_j: torch.Tensor,
    offsets: torch.Tensor,
    alphas: torch.Tensor,
    powers: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    j_max = prompt_of_chunk.numel()
    n_scales = alphas.numel()
    d_feat = feat_seq.size(1)
    out = torch.empty(
        (j_max, n_scales, d_feat), device=feat_seq.device, dtype=torch.float32
    )
    grid = (j_max, n_scales, triton.cdiv(d_feat, 64))
    _chunk_end_kernel[grid](
        feat_seq,
        prompt_of_chunk,
        start_local,
        len_j,
        offsets,
        alphas,
        powers,
        out,
        D=d_feat,
        T1=powers.size(1),
        C=chunk_size,
        BLOCK_C=triton.next_power_of_2(chunk_size),
        BLOCK_D=64,
        num_warps=4,
    )
    return out


def sampled_output(
    feat_seq: torch.Tensor,
    sample_idx: torch.Tensor,
    prompt_idx: torch.Tensor,
    local_pos: torch.Tensor,
    chunk_id: torch.Tensor,
    start_local: torch.Tensor,
    offsets: torch.Tensor,
    carry: torch.Tensor,
    alphas: torch.Tensor,
    powers: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    m = sample_idx.numel()
    n_scales = alphas.numel()
    d_feat = feat_seq.size(1)
    out = torch.empty((m, n_scales, d_feat), device=feat_seq.device, dtype=feat_seq.dtype)
    grid = (m, n_scales, triton.cdiv(d_feat, 64))
    _sampled_output_kernel[grid](
        feat_seq,
        sample_idx,
        prompt_idx,
        local_pos,
        chunk_id,
        start_local,
        offsets,
        carry,
        alphas,
        powers,
        out,
        D=d_feat,
        T1=powers.size(1),
        C=chunk_size,
        BLOCK_C=triton.next_power_of_2(chunk_size),
        BLOCK_D=64,
        num_warps=4,
    )
    return out
