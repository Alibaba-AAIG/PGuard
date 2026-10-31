from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only and minimal test environments use the PyTorch fallback.
    triton = None
    tl = None


_NUM_SCALES = 4
_BLOCK_D = 64


if triton is not None:
    _TRITON_NUM_SCALES = tl.constexpr(_NUM_SCALES)
    _TRITON_DOT_SCALES = tl.constexpr(4 * _NUM_SCALES)

    @triton.jit
    def _retention_power(retention, distance):
        # pow(0, 0) is 1 in torch; log(0) * 0 would otherwise produce NaN.
        return tl.where(
            distance == 0,
            1.0,
            tl.exp(tl.log(retention) * distance.to(tl.float32)),
        )

    @triton.jit
    def _chunk_summary_kernel(
        feat_ptr,
        seq_lens_ptr,
        offsets_ptr,
        prompt_of_chunk_ptr,
        local_chunk_ptr,
        alphas_ptr,
        summary_ptr,
        feat_stride_t: tl.constexpr,
        feat_stride_d: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Stage 1: BF16-rounded EMA contribution at every chunk boundary."""
        chunk_id = tl.program_id(0)
        d_block = tl.program_id(1)
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        token_in_chunk = tl.arange(0, C)
        scale = tl.arange(0, _TRITON_DOT_SCALES)

        prompt = tl.load(prompt_of_chunk_ptr + chunk_id)
        local_chunk = tl.load(local_chunk_ptr + chunk_id)
        seq_len = tl.load(seq_lens_ptr + prompt)
        prompt_offset = tl.load(offsets_ptr + prompt)
        chunk_start = local_chunk * C
        chunk_len = tl.minimum(seq_len - chunk_start, C)

        token = prompt_offset + chunk_start + token_in_chunk
        feat = tl.load(
            feat_ptr
            + token[:, None] * feat_stride_t
            + d[None, :] * feat_stride_d,
            mask=(token_in_chunk[:, None] < chunk_len) & (d[None, :] < D),
            other=0.0,
        )

        alpha = tl.load(
            alphas_ptr + scale,
            mask=scale < _TRITON_NUM_SCALES,
            other=0.0,
        ).to(tl.float32)
        retention = 1.0 - alpha
        distance = chunk_len - 1 - token_in_chunk
        valid_weight = (scale[:, None] < _TRITON_NUM_SCALES) & (
            token_in_chunk[None, :] < chunk_len
        )
        weight = alpha[:, None] * _retention_power(
            retention[:, None], distance[None, :]
        )
        weight = tl.where(valid_weight, weight, 0.0).to(tl.bfloat16)
        within = tl.dot(weight, feat, out_dtype=tl.float32)

        out_offset = (
            chunk_id * _TRITON_NUM_SCALES * D
            + scale[:, None] * D
            + d[None, :]
        )
        tl.store(
            summary_ptr + out_offset,
            within.to(tl.bfloat16),
            mask=(scale[:, None] < _TRITON_NUM_SCALES) & (d[None, :] < D),
        )


    @triton.jit
    def _prompt_carry_kernel(
        summary_ptr,
        ema_old_ptr,
        seq_lens_ptr,
        n_chunks_ptr,
        chunk_offsets_ptr,
        alphas_ptr,
        carry_ptr,
        ema_stride_b: tl.constexpr,
        ema_stride_s: tl.constexpr,
        ema_stride_d: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Stage 2: sequential FP32 carry at each prompt's chunk starts."""
        prompt = tl.program_id(0)
        d_block = tl.program_id(1)
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        scale = tl.arange(0, _TRITON_NUM_SCALES)
        mask = d[None, :] < D

        alpha = tl.load(alphas_ptr + scale).to(tl.float32)
        retention = 1.0 - alpha
        carry = tl.load(
            ema_old_ptr
            + prompt * ema_stride_b
            + scale[:, None] * ema_stride_s
            + d[None, :] * ema_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        seq_len = tl.load(seq_lens_ptr + prompt)
        num_chunks = tl.load(n_chunks_ptr + prompt)
        first_chunk = tl.load(chunk_offsets_ptr + prompt)

        local_chunk = 0
        while local_chunk < num_chunks:
            chunk_id = first_chunk + local_chunk
            offset = (
                chunk_id * _TRITON_NUM_SCALES * D
                + scale[:, None] * D
                + d[None, :]
            )
            tl.store(carry_ptr + offset, carry, mask=mask)
            summary = tl.load(summary_ptr + offset, mask=mask, other=0.0).to(
                tl.float32
            )
            chunk_len = tl.minimum(seq_len - local_chunk * C, C)
            decay = tl.exp(
                tl.log(retention) * chunk_len.to(tl.float32)
            )
            carry = summary + decay[:, None] * carry
            local_chunk += 1


    @triton.jit
    def _sampled_output_kernel(
        feat_ptr,
        offsets_ptr,
        prompt_idx_ptr,
        local_pos_ptr,
        sample_idx_ptr,
        chunk_offsets_ptr,
        alphas_ptr,
        carry_ptr,
        output_ptr,
        feat_stride_t: tl.constexpr,
        feat_stride_d: tl.constexpr,
        sample_stride: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Stage 3: BF16-rounded local dot plus the FP32 incoming carry."""
        sample_row = tl.program_id(0)
        d_block = tl.program_id(1)
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        token_in_chunk = tl.arange(0, C)
        scale = tl.arange(0, _TRITON_DOT_SCALES)

        sample = tl.load(sample_idx_ptr + sample_row * sample_stride)
        prompt = tl.load(prompt_idx_ptr + sample)
        position = tl.load(local_pos_ptr + sample)
        local_chunk = position // C
        pos_in_chunk = position - local_chunk * C
        chunk_id = tl.load(chunk_offsets_ptr + prompt) + local_chunk
        prompt_offset = tl.load(offsets_ptr + prompt)
        chunk_start = prompt_offset + local_chunk * C

        token = chunk_start + token_in_chunk
        feat = tl.load(
            feat_ptr
            + token[:, None] * feat_stride_t
            + d[None, :] * feat_stride_d,
            mask=(token_in_chunk[:, None] <= pos_in_chunk) & (d[None, :] < D),
            other=0.0,
        )

        alpha = tl.load(
            alphas_ptr + scale,
            mask=scale < _TRITON_NUM_SCALES,
            other=0.0,
        ).to(tl.float32)
        retention = 1.0 - alpha
        distance = pos_in_chunk - token_in_chunk
        valid_weight = (scale[:, None] < _TRITON_NUM_SCALES) & (
            token_in_chunk[None, :] <= pos_in_chunk
        )
        weight = alpha[:, None] * _retention_power(
            retention[:, None], distance[None, :]
        )
        weight = tl.where(valid_weight, weight, 0.0).to(tl.bfloat16)
        within = tl.dot(weight, feat, out_dtype=tl.float32)
        # Match torch.einsum(...)->BF16 followed by .to(FP32) before carry/add.
        within = within.to(tl.bfloat16).to(tl.float32)

        real_scale = scale < _TRITON_NUM_SCALES
        carry_offset = (
            chunk_id * _TRITON_NUM_SCALES * D
            + scale[:, None] * D
            + d[None, :]
        )
        carry = tl.load(
            carry_ptr + carry_offset,
            mask=real_scale[:, None] & (d[None, :] < D),
            other=0.0,
        )
        decay = tl.exp(
            tl.log(retention) * (pos_in_chunk + 1).to(tl.float32)
        )
        output = within + decay[:, None] * carry
        output_offset = (
            sample_row * _TRITON_NUM_SCALES * D
            + scale[:, None] * D
            + d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            output.to(tl.bfloat16),
            mask=real_scale[:, None] & (d[None, :] < D),
        )


def can_use_fused_sampled_ema(
    feat_seq: torch.Tensor,
    ema_old: torch.Tensor,
    ema_alphas: torch.Tensor,
    sample_idx: torch.Tensor,
    chunk_size: int,
) -> bool:
    """Return whether the specialized production-shape kernel can run."""
    return bool(
        triton is not None
        and feat_seq.device.type == "cuda"
        and ema_old.device == feat_seq.device
        and ema_alphas.device == feat_seq.device
        and sample_idx.device == feat_seq.device
        and feat_seq.dtype == torch.bfloat16
        and ema_old.dtype == torch.bfloat16
        and ema_alphas.dtype == torch.float32
        and feat_seq.dim() == 2
        and ema_old.dim() == 3
        and ema_alphas.dim() == 1
        and sample_idx.dim() == 1
        and ema_old.size(0) > 0
        and ema_old.size(1) == _NUM_SCALES
        and ema_old.size(2) == feat_seq.size(1)
        and ema_alphas.numel() == _NUM_SCALES
        and feat_seq.stride(-1) == 1
        and ema_old.stride(-1) == 1
        and chunk_size in (64, 128)
        and feat_seq.size(0) > 0
        and feat_seq.size(1) > 0
        and sample_idx.numel() > 0
    )


def fused_sampled_causal_ema(
    feat_seq: torch.Tensor,
    seq_lens: torch.Tensor,
    offsets: torch.Tensor,
    ema_old: torch.Tensor,
    prompt_idx: torch.Tensor,
    local_pos: torch.Tensor,
    sample_idx: torch.Tensor,
    ema_alphas: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run chunk-summary, prompt-carry, and sampled-output Triton kernels.

    Callers must check :func:`can_use_fused_sampled_ema` first.  No synchronization,
    scalar extraction, autotuning, or alternate-stream work is performed here.
    """
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    T, D = feat_seq.shape
    B = seq_lens.numel()
    M = sample_idx.numel()
    C = int(chunk_size)
    J_max = (T - 1) // C + B

    seq_lens = seq_lens.contiguous()
    offsets = offsets.contiguous()
    prompt_idx = prompt_idx.contiguous()
    local_pos = local_pos.contiguous()
    sample_idx = sample_idx.contiguous()
    alphas = ema_alphas.contiguous()

    n_chunks = torch.div(seq_lens + C - 1, C, rounding_mode="floor")
    chunk_offsets = torch.cumsum(n_chunks, dim=0) - n_chunks
    local_chunk = torch.div(local_pos, C, rounding_mode="floor")
    chunk_id = chunk_offsets[prompt_idx] + local_chunk

    # J_max is a host-known upper bound.  At most B-1 trailing rows are unused.
    prompt_of_chunk = torch.zeros(J_max, dtype=torch.long, device=feat_seq.device)
    local_chunk_of_chunk = torch.zeros_like(prompt_of_chunk)
    prompt_of_chunk.index_copy_(0, chunk_id, prompt_idx)
    local_chunk_of_chunk.index_copy_(0, chunk_id, local_chunk)

    summary = torch.empty(
        (J_max, _NUM_SCALES, D), dtype=torch.bfloat16, device=feat_seq.device
    )
    carry = torch.empty(
        (J_max, _NUM_SCALES, D), dtype=torch.float32, device=feat_seq.device
    )

    d_blocks = triton.cdiv(D, _BLOCK_D)
    _chunk_summary_kernel[(J_max, d_blocks)](
        feat_seq,
        seq_lens,
        offsets,
        prompt_of_chunk,
        local_chunk_of_chunk,
        alphas,
        summary,
        feat_seq.stride(0),
        feat_seq.stride(1),
        D=D,
        C=C,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )
    _prompt_carry_kernel[(B, d_blocks)](
        summary,
        ema_old,
        seq_lens,
        n_chunks,
        chunk_offsets,
        alphas,
        carry,
        ema_old.stride(0),
        ema_old.stride(1),
        ema_old.stride(2),
        D=D,
        C=C,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )
    del summary
    output = torch.empty(
        (M, _NUM_SCALES, D), dtype=torch.bfloat16, device=feat_seq.device
    )
    _sampled_output_kernel[(M, d_blocks)](
        feat_seq,
        offsets,
        prompt_idx,
        local_pos,
        sample_idx,
        chunk_offsets,
        alphas,
        carry,
        output,
        feat_seq.stride(0),
        feat_seq.stride(1),
        sample_idx.stride(0),
        D=D,
        C=C,
        BLOCK_D=_BLOCK_D,
        num_warps=4,
    )
    return output
