from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import (
    PGUARD_SCORE_KMAX_SIGMA as _SCORE_KMAX_SIGMA,
    PGUARD_SEQ_SAMPLED_SCAN_CHUNK,
)
from . import sampled_ema_kernel

# Full scan and DSV/Qwen configured sampled scans retain their fixed chunk sizes.
_FULL_SCAN_CHUNK = 128
_SAMPLED_SCAN_CHUNK_DENSE = 48
_SAMPLED_SCAN_CHUNK_SPARSE = 64


def _sampled_scan_chunk(total_tokens: int, sampled_tokens: int) -> int:
    """GLM adaptive scan: choose from host-known shapes, never device values."""
    return (
        _SAMPLED_SCAN_CHUNK_SPARSE
        if sampled_tokens * 16 <= total_tokens
        else _SAMPLED_SCAN_CHUNK_DENSE
    )


class HistoryFeatureBuffer(nn.Module):

    def __init__(
        self,
        max_req: int,
        d_feat: int,
        window_n: int,
        ema_scales=(200.0, 150.0, 100.0, 50.0),
        dtype: torch.dtype = torch.bfloat16,
        num_classes: int = 0,
        sampled_scan_chunk: Optional[int] = PGUARD_SEQ_SAMPLED_SCAN_CHUNK,
        fused_sampled_ema: bool = False,
    ) -> None:
        super().__init__()
        assert max_req > 0, f"max_req must be >0, got {max_req}"
        assert window_n >= 0, f"window_n must be >=0, got {window_n}"
        assert num_classes >= 0, f"num_classes must be >=0, got {num_classes}"
        assert len(ema_scales) > 0, "ema_scales must be non-empty"
        # None opts GLM into adaptive C48/64; omitted arguments retain DSV/Qwen defaults.
        if sampled_scan_chunk not in (None, 16, 32, 64, 128, 256):
            raise ValueError(
                "sampled_scan_chunk must be None (adaptive) or one of 16,32,64,128,256, "
                f"got {sampled_scan_chunk}"
            )
        self.max_req = max_req
        self.d_feat = d_feat
        self.N = window_n
        self.num_classes = num_classes
        self.sampled_scan_chunk = sampled_scan_chunk
        self.fused_sampled_ema = bool(fused_sampled_ema)
        self.ema_scales = tuple(float(t) for t in sorted(ema_scales, reverse=True))
        self.n_scales = len(self.ema_scales)
        self.register_buffer(
            "ema_alphas",
            torch.tensor([1.0 - math.exp(-1.0 / tau) for tau in self.ema_scales], dtype=torch.float32),
            persistent=False,
        )  # [n_scales]

        self.register_buffer(
            "hist_buf",
            torch.zeros(max_req, max(window_n, 0), d_feat, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "hist_ema",
            torch.zeros(max_req, self.n_scales, d_feat, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "hist_head",
            torch.zeros(max_req, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "hist_step",
            torch.zeros(max_req, dtype=torch.int32),
            persistent=False,
        )
        if num_classes > 0:
            self.register_buffer(
                "hist_risk_max",
                torch.full((max_req, num_classes), float("-inf"), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.hist_risk_max = None
        self._sampled_power_cache = {}

    def _apply(self, fn):
        result = super()._apply(fn)
        self._sampled_power_cache.clear()
        return result

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset_slots(self, slot_ids: torch.Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        slot_ids = slot_ids.to(self.hist_ema.device, dtype=torch.long)
        self.hist_ema.index_fill_(0, slot_ids, 0.0)
        self.hist_head.index_fill_(0, slot_ids, 0)
        self.hist_step.index_fill_(0, slot_ids, 0)
        if self.N > 0:
            zeros = self.hist_buf.new_zeros(slot_ids.numel(), self.N, self.d_feat)
            self.hist_buf.index_copy_(0, slot_ids, zeros)
        if self.hist_risk_max is not None:
            self.hist_risk_max.index_fill_(0, slot_ids, float("-inf"))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def reset_slots_masked(self, slot_ids: torch.Tensor, fresh: torch.Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        slot_ids = slot_ids.to(self.hist_ema.device, dtype=torch.long)
        keep = (~fresh).to(self.hist_ema.dtype)                              # [B]
        self.hist_ema[slot_ids] = self.hist_ema[slot_ids] * keep.view(-1, 1, 1)
        self.hist_head[slot_ids] = self.hist_head[slot_ids] * (~fresh).to(self.hist_head.dtype)
        self.hist_step[slot_ids] = self.hist_step[slot_ids] * (~fresh).to(self.hist_step.dtype)
        if self.N > 0:
            self.hist_buf[slot_ids] = (
                self.hist_buf[slot_ids] * keep.view(-1, 1, 1).to(self.hist_buf.dtype)
            )
        if self.hist_risk_max is not None:
            self.hist_risk_max[slot_ids] = torch.where(
                fresh.view(-1, 1),
                torch.full_like(self.hist_risk_max[slot_ids], float("-inf")),
                self.hist_risk_max[slot_ids],
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def update_risk_max(self, slot_ids: torch.Tensor, chunk_max: torch.Tensor) -> torch.Tensor:
        assert self.hist_risk_max is not None, (
            "hist_risk_max 未初始化（构造 HistoryFeatureBuffer 时 num_classes=0）"
        )
        slot_ids = slot_ids.to(self.hist_risk_max.device, dtype=torch.long)
        cur = self.hist_risk_max.index_select(0, slot_ids)              # [B, C]
        updated = torch.maximum(cur, chunk_max.to(cur.dtype))           # [B, C]
        self.hist_risk_max.index_copy_(0, slot_ids, updated)
        return updated

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def read_and_update(
        self,
        slot_ids: torch.Tensor,
        feat_seq: torch.Tensor,
        is_extend: bool,
        extend_prefix_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert feat_seq.dim() == 2 and feat_seq.size(-1) == self.d_feat, (
            f"feat_seq shape expected (B, {self.d_feat}), got {tuple(feat_seq.shape)}"
        )
        slot_ids = slot_ids.to(self.hist_ema.device, dtype=torch.long)
        B = slot_ids.size(0)
        assert feat_seq.size(0) == B, (
            f"feat_seq batch {feat_seq.size(0)} != slot_ids batch {B}"
        )

        if is_extend and extend_prefix_lens is not None:
            fresh = extend_prefix_lens == 0  # [B] bool
            if fresh.any():
                self.reset_slots(slot_ids[fresh])

        feat_seq = feat_seq.to(self.hist_ema.dtype)

        ema_prev = self.hist_ema.index_select(0, slot_ids)          # [B, n_scales, D]

        a = self.ema_alphas.to(feat_seq.dtype).view(1, -1, 1)         # [1, n_scales, 1]
        ema_new = a * feat_seq.unsqueeze(1) + (1.0 - a) * ema_prev    # [B, n_scales, D]
        self.hist_ema.index_copy_(0, slot_ids, ema_new)

        step_prev = self.hist_step.index_select(0, slot_ids).to(torch.long)  # [B]

        if self.N > 0:
            head_prev = self.hist_head.index_select(0, slot_ids).to(torch.long)  # [B]

            self.hist_buf[slot_ids, head_prev] = feat_seq  # inplace

            hist_raw = self.hist_buf.index_select(0, slot_ids)  # [B, N, D]
            new_head = (head_prev + 1) % self.N
            # hist_seq[b, k] = hist_raw[b, (new_head[b] + k) % N]
            k_range = torch.arange(self.N, device=slot_ids.device)  # [N]
            gather_idx = (new_head.unsqueeze(-1) + k_range.unsqueeze(0)) % self.N  # [B, N]
            gather_idx_expanded = gather_idx.unsqueeze(-1).expand(-1, -1, self.d_feat)  # [B, N, D]
            hist_seq = torch.gather(hist_raw, 1, gather_idx_expanded)  # [B, N, D]

            self.hist_head.index_copy_(0, slot_ids, new_head.to(torch.int32))

            step_new = step_prev + 1
            valid_len = torch.clamp(step_new, max=self.N)           # [B]
            # mask[b, k] = k >= N - valid_len[b]
            hist_mask = k_range.unsqueeze(0) >= (self.N - valid_len.unsqueeze(-1))  # [B, N]
        else:
            hist_seq = feat_seq.new_zeros(B, 0, self.d_feat)
            hist_mask = torch.zeros(B, 0, dtype=torch.bool, device=slot_ids.device)

        step_new_i32 = (step_prev + 1).to(torch.int32)
        self.hist_step.index_copy_(0, slot_ids, step_new_i32)

        return hist_seq, hist_mask, ema_new

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def prefill_update(
        self,
        slot_ids: torch.Tensor,
        feat_seq: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: Optional[torch.Tensor],
        per_token: bool,
        sample_idx: Optional[torch.Tensor] = None,
        eps: float = 1e-2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert feat_seq.dim() == 2 and feat_seq.size(-1) == self.d_feat, (
            f"feat_seq shape expected (T, {self.d_feat}), got {tuple(feat_seq.shape)}"
        )
        device = self.hist_ema.device
        slot_ids = slot_ids.to(device, dtype=torch.long)
        feat_seq = feat_seq.to(device)
        B = slot_ids.size(0)
        N = self.N
        D = self.d_feat

        if extend_prefix_lens is not None:
            self.reset_slots_masked(slot_ids, extend_prefix_lens.to(device) == 0)

        ema_old = self.hist_ema.index_select(0, slot_ids)                    # [B, D]
        head_old = self.hist_head.index_select(0, slot_ids).to(torch.long)   # [B]
        step_old = self.hist_step.index_select(0, slot_ids).to(torch.long)   # [B]
        prev_ring = (
            self.hist_buf.index_select(0, slot_ids) if (per_token and N > 0) else None
        )  # [B, N, D]

        seq_lens = seq_lens.to(device, dtype=torch.long)
        T = feat_seq.size(0)
        prompt_idx = torch.repeat_interleave(
            torch.arange(B, device=device), seq_lens, output_size=T
        )  # [T]
        offsets = torch.cumsum(seq_lens, dim=0) - seq_lens                              # [B]
        t_arange = torch.arange(T, device=device)                                        # [T]
        local_pos = t_arange - offsets[prompt_idx]                                       # [T]
        dist_from_end = seq_lens[prompt_idx] - 1 - local_pos                             # [T]

        a = self.ema_alphas.to(torch.float32)                                # [n_scales]
        one_minus_a = 1.0 - a                                                # [n_scales]
        w = a.view(-1, 1) * torch.pow(
            one_minus_a.view(-1, 1), dist_from_end.view(1, -1).to(torch.float32)
        )                                                                    # [n_scales, T]
        P = (
            prompt_idx.view(1, -1) == torch.arange(B, device=device).view(-1, 1)
        ).to(torch.float32)                                                  # [B, T] one-hot
        Wp = w.unsqueeze(1) * P.unsqueeze(0)                                 # [n_scales, B, T]
        ema_within = torch.matmul(Wp.to(feat_seq.dtype), feat_seq)           # [n_scales, B, D]
        decay = torch.pow(
            one_minus_a.view(-1, 1), seq_lens.view(1, -1).to(torch.float32)
        )                                                                    # [n_scales, B]
        ema_old_p = ema_old.permute(1, 0, 2).to(torch.float32)               # [n_scales, B, D]
        ema_final = (ema_within.to(torch.float32) + decay.unsqueeze(-1) * ema_old_p).permute(1, 0, 2)  # [B, n_scales, D]

        if per_token:
            if sample_idx is not None:
                ema_out = self._sampled_causal_ema(
                    feat_seq, seq_lens, offsets, ema_old, prompt_idx, local_pos, sample_idx,
                )                                                                # [M, n_scales, D]
            else:
                ema_out = self._multi_scale_causal_ema(
                    feat_seq, seq_lens, offsets, ema_old, prompt_idx, local_pos, eps=eps,
                )                                                                # [T, n_scales, D]
        else:
            ema_out = ema_final.to(self.hist_ema.dtype)                      # [B, n_scales, D]

        self.hist_ema.index_copy_(0, slot_ids, ema_final.to(self.hist_ema.dtype))
        if N > 0:
            in_window = dist_from_end < N                                                # [T]
            ring_pos = (head_old[prompt_idx] + local_pos) % N                            # [T]
            sel = in_window.nonzero(as_tuple=True)[0]
            self.hist_buf[slot_ids[prompt_idx[sel]], ring_pos[sel]] = (
                feat_seq[sel].to(self.hist_buf.dtype)
            )
            head_new = (head_old + seq_lens) % N
        else:
            head_new = head_old
        step_new = step_old + seq_lens
        self.hist_head.index_copy_(0, slot_ids, head_new.to(torch.int32))
        self.hist_step.index_copy_(0, slot_ids, step_new.to(torch.int32))

        # 7) hist_seq / hist_mask
        if per_token:
            if N > 0:
                k_range = torch.arange(N, device=device)                                 # [N]
                src = t_arange.unsqueeze(1) - (N - 1) + k_range.unsqueeze(0)             # [T, N]
                local_src = local_pos.unsqueeze(1) - (N - 1) + k_range.unsqueeze(0)      # [T, N]
                within = local_src >= 0                                                  # [T, N]
                hist_seq = feat_seq[src.clamp(min=0)] * within.unsqueeze(-1).to(feat_seq.dtype)
                ring_valid = local_src >= -step_old[prompt_idx].unsqueeze(1)             # [T, N]
                need_ring = (~within) & ring_valid                                       # [T, N]
                if need_ring.any():
                    ring_idx = (head_old[prompt_idx].unsqueeze(1) + local_src) % N       # [T, N]
                    hist_ring = prev_ring[
                        prompt_idx.unsqueeze(1).expand(-1, N), ring_idx
                    ]                                                                    # [T, N, D]
                    hist_seq = hist_seq + hist_ring * need_ring.unsqueeze(-1).to(hist_seq.dtype)
                hist_mask = within | ring_valid                                          # [T, N]
            else:
                hist_seq = feat_seq.new_zeros(T, 0, D)
                hist_mask = torch.zeros(T, 0, dtype=torch.bool, device=device)
        else:
            if N > 0:
                hist_raw = self.hist_buf.index_select(0, slot_ids)                       # [B, N, D]
                k_range = torch.arange(N, device=device)                                 # [N]
                gather_idx = (head_new.unsqueeze(-1) + k_range.unsqueeze(0)) % N         # [B, N]
                hist_seq = torch.gather(
                    hist_raw, 1, gather_idx.unsqueeze(-1).expand(-1, -1, D)
                )                                                                        # [B, N, D]
                valid_len = torch.clamp(step_new, max=N)                                 # [B]
                hist_mask = k_range.unsqueeze(0) >= (N - valid_len.unsqueeze(-1))        # [B, N]
            else:
                hist_seq = feat_seq.new_zeros(B, 0, D)
                hist_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)

        if per_token and sample_idx is not None:
            hist_seq = hist_seq[sample_idx]
            hist_mask = hist_mask[sample_idx]

        return hist_seq, hist_mask, ema_out

    def _multi_scale_causal_ema(
        self,
        feat_seq: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        ema_old: torch.Tensor,
        prompt_idx: torch.Tensor,
        local_pos: torch.Tensor,
        eps: float = 1e-2,
    ) -> torch.Tensor:
        device = feat_seq.device
        D = self.d_feat
        T = feat_seq.size(0)
        B = seq_lens.size(0)
        n = self.n_scales
        C = _FULL_SCAN_CHUNK
        a = self.ema_alphas.to(torch.float32)                            # [n]
        r = 1.0 - a                                                      # [n]

        J_max = T // C + B
        n_chunks = (seq_lens + C - 1) // C                               # [B]
        chunk_offset = torch.cumsum(n_chunks, dim=0) - n_chunks          # [B]
        local_chunk = local_pos // C                                     # [T]
        pos_in_chunk = local_pos - local_chunk * C                       # [T]
        chunk_id = chunk_offset[prompt_idx] + local_chunk                # [T]

        prompt_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        prompt_of_chunk.index_copy_(0, chunk_id, prompt_idx)
        lchunk_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        lchunk_of_chunk.index_copy_(0, chunk_id, local_chunk)
        valid_chunk = torch.zeros(J_max, dtype=torch.bool, device=device)
        valid_chunk.index_fill_(0, chunk_id, True)

        seq_lens_chunk = seq_lens[prompt_of_chunk]                       # [J]
        start_local = lchunk_of_chunk * C                                # [J]
        len_j = torch.clamp(seq_lens_chunk - start_local, max=C)         # [J]
        len_j = len_j * valid_chunk

        k_range = torch.arange(C, device=device)
        tok_local = start_local.unsqueeze(1) + k_range.unsqueeze(0)      # [J, C]
        in_prompt = tok_local < seq_lens_chunk.unsqueeze(1)              # [J, C]
        tok_idx = offsets[prompt_of_chunk].unsqueeze(1) + tok_local      # [J, C]
        feat_pad = feat_seq[tok_idx.clamp(min=0, max=T - 1)]             # [J, C, D]
        dist = k_range.unsqueeze(1) - k_range.unsqueeze(0)               # [C, C] (i,k)
        causal = dist >= 0
        W_intra = a.view(-1, 1, 1) * torch.pow(
            r.view(-1, 1, 1), dist.clamp(min=0).to(torch.float32)
        ) * causal                                                       # [n, C, C]

        out = torch.zeros(T, n, D, dtype=torch.float32, device=device)
        w = torch.zeros(J_max, n, D, dtype=torch.float32, device=device)
        end_pos = (len_j - 1).clamp(min=0)                               # [J]
        j_arange = torch.arange(J_max, device=device)
        for i in range(n):
            W_i = W_intra[i].unsqueeze(0) * in_prompt.unsqueeze(1)       # [J, C, C]
            ew = torch.matmul(W_i.to(feat_pad.dtype), feat_pad)          # [J, C, D]
            out[:, i] = ew[chunk_id, pos_in_chunk]                       # [T, D]
            w[:, i] = ew[j_arange, end_pos] * valid_chunk.unsqueeze(1)

        end_of = start_local + len_j                                     # [J]
        dist_c = start_local.unsqueeze(1) - end_of.unsqueeze(0)          # [J, J] (j,j')
        same_prompt = prompt_of_chunk.unsqueeze(1) == prompt_of_chunk.unsqueeze(0)
        prop = same_prompt & (dist_c >= 0)                               # [J, J]
        P = torch.pow(
            r.view(-1, 1, 1), dist_c.clamp(min=0).to(torch.float32).unsqueeze(0)
        ) * prop                                                         # [n, J, J]
        S = torch.matmul(P, w.permute(1, 0, 2)).permute(1, 0, 2)         # [J, n, D]
        decay_init = torch.pow(
            r.unsqueeze(0), start_local.to(torch.float32).unsqueeze(1)
        )                                                                # [J, n]
        S = S + decay_init.unsqueeze(-1) * ema_old[prompt_of_chunk].to(torch.float32)

        # ---- combine：ema[t] = within[chunk,pos] + r^(pos+1)·S[chunk] ----
        decay_t = torch.pow(
            r.unsqueeze(0), (pos_in_chunk + 1).to(torch.float32).unsqueeze(1)
        )                                                                # [T, n]
        out = out + decay_t.unsqueeze(-1) * S[chunk_id]
        return out.to(feat_seq.dtype)

    def _sampled_causal_ema(
        self,
        feat_seq: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        ema_old: torch.Tensor,
        prompt_idx: torch.Tensor,
        local_pos: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.fused_sampled_ema and self.sampled_scan_chunk is None:
            from .seq_ema_triton import can_use_fused_sampled_ema

            if can_use_fused_sampled_ema(
                feat_seq, seq_lens, offsets, ema_old, prompt_idx, local_pos,
                sample_idx, self.ema_alphas,
                _sampled_scan_chunk(feat_seq.size(0), sample_idx.numel()),
            ):
                return self._sampled_causal_ema_fused(
                    feat_seq, seq_lens, offsets, ema_old, prompt_idx, local_pos, sample_idx
                )
        elif self.fused_sampled_ema and sampled_ema_kernel.can_use_fused_sampled_ema(
            feat_seq,
            ema_old,
            self.ema_alphas,
            sample_idx,
            self.sampled_scan_chunk,
        ):
            return sampled_ema_kernel.fused_sampled_causal_ema(
                feat_seq=feat_seq,
                seq_lens=seq_lens,
                offsets=offsets,
                ema_old=ema_old,
                prompt_idx=prompt_idx,
                local_pos=local_pos,
                sample_idx=sample_idx,
                ema_alphas=self.ema_alphas,
                chunk_size=self.sampled_scan_chunk,
            )

        device = feat_seq.device
        D = self.d_feat
        T = feat_seq.size(0)
        B = seq_lens.size(0)
        n = self.n_scales
        C = (
            self.sampled_scan_chunk
            if self.sampled_scan_chunk is not None
            else _sampled_scan_chunk(T, sample_idx.numel())
        )
        a = self.ema_alphas.to(torch.float32)                            # [n]
        r = 1.0 - a                                                      # [n]

        J_max = T // C + B
        n_chunks = (seq_lens + C - 1) // C                               # [B]
        chunk_offset = torch.cumsum(n_chunks, dim=0) - n_chunks          # [B]
        local_chunk = local_pos // C                                     # [T]
        pos_in_chunk = local_pos - local_chunk * C                       # [T]
        chunk_id = chunk_offset[prompt_idx] + local_chunk                # [T]

        prompt_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        prompt_of_chunk.index_copy_(0, chunk_id, prompt_idx)
        lchunk_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        lchunk_of_chunk.index_copy_(0, chunk_id, local_chunk)
        valid_chunk = torch.zeros(J_max, dtype=torch.bool, device=device)
        valid_chunk.index_fill_(0, chunk_id, True)

        seq_lens_chunk = seq_lens[prompt_of_chunk]                       # [J]
        start_local = lchunk_of_chunk * C                                # [J]
        len_j = torch.clamp(seq_lens_chunk - start_local, max=C)         # [J]
        len_j = len_j * valid_chunk

        k_range = torch.arange(C, device=device)
        tok_local = start_local.unsqueeze(1) + k_range.unsqueeze(0)      # [J, C]
        in_prompt = tok_local < seq_lens_chunk.unsqueeze(1)              # [J, C]
        tok_idx = offsets[prompt_of_chunk].unsqueeze(1) + tok_local      # [J, C]
        feat_pad = feat_seq[tok_idx.clamp(min=0, max=T - 1)]             # [J, C, D]
        dist_end = (len_j - 1).unsqueeze(1) - k_range.unsqueeze(0)       # [J, C]
        w_end = a.view(1, -1, 1) * torch.pow(
            r.view(1, -1, 1), dist_end.clamp(min=0).to(torch.float32).unsqueeze(1)
        ) * (dist_end >= 0).unsqueeze(1)                                 # [J, n, C]
        within_end = torch.einsum('jnc,jcd->jnd', w_end.to(feat_pad.dtype), feat_pad)
        within_end = within_end.to(torch.float32) * valid_chunk.unsqueeze(-1).unsqueeze(-1)

        end_of = start_local + len_j                                     # [J]
        dist_c = start_local.unsqueeze(1) - end_of.unsqueeze(0)          # [J, J]
        same_prompt = prompt_of_chunk.unsqueeze(1) == prompt_of_chunk.unsqueeze(0)
        prop = same_prompt & (dist_c >= 0)                               # [J, J]
        P = torch.pow(
            r.view(-1, 1, 1), dist_c.clamp(min=0).to(torch.float32).unsqueeze(0)
        ) * prop                                                         # [n, J, J]
        S = torch.matmul(P, within_end.permute(1, 0, 2)).permute(1, 0, 2)  # [J, n, D]
        decay_init = torch.pow(
            r.unsqueeze(0), start_local.to(torch.float32).unsqueeze(1)
        )                                                                # [J, n]
        S = S + decay_init.unsqueeze(-1) * ema_old[prompt_of_chunk].to(torch.float32)

        s_chunk = chunk_id[sample_idx]                                   # [M]
        s_pos = pos_in_chunk[sample_idx]                                 # [M]
        s_prompt = prompt_idx[sample_idx]                                # [M]
        win_start = offsets[s_prompt] + start_local[s_chunk]             # [M]
        win_idx = win_start.unsqueeze(1) + k_range.unsqueeze(0)          # [M, C]
        feat_w = feat_seq[win_idx.clamp(min=0, max=T - 1)]               # [M, C, D]
        dist_s = s_pos.unsqueeze(1) - k_range.unsqueeze(0)               # [M, C]
        w_s = a.view(1, -1, 1) * torch.pow(
            r.view(1, -1, 1), dist_s.clamp(min=0).to(torch.float32).unsqueeze(1)
        ) * (dist_s >= 0).unsqueeze(1)                                   # [M, n, C]
        within_s = torch.einsum('mnc,mcd->mnd', w_s.to(feat_w.dtype), feat_w)
        within_s = within_s.to(torch.float32)
        decay_s = torch.pow(
            r.unsqueeze(0), (s_pos + 1).to(torch.float32).unsqueeze(1)
        )                                                                # [M, n]
        out = within_s + decay_s.unsqueeze(-1) * S[s_chunk].to(torch.float32)
        return out.to(feat_seq.dtype)

    def _sampled_causal_ema_fused(
        self,
        feat_seq: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        ema_old: torch.Tensor,
        prompt_idx: torch.Tensor,
        local_pos: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        from .seq_ema_triton import sampled_chunk_end, sampled_output

        device = feat_seq.device
        T = feat_seq.size(0)
        B = seq_lens.size(0)
        C = _sampled_scan_chunk(T, sample_idx.numel())
        a = self.ema_alphas.to(device=device, dtype=torch.float32)
        r = 1.0 - a

        J_max = T // C + B
        n_chunks = (seq_lens + C - 1) // C
        chunk_offset = torch.cumsum(n_chunks, dim=0) - n_chunks
        local_chunk = local_pos // C
        chunk_id = chunk_offset[prompt_idx] + local_chunk

        prompt_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        prompt_of_chunk.index_copy_(0, chunk_id, prompt_idx)
        lchunk_of_chunk = torch.zeros(J_max, dtype=torch.long, device=device)
        lchunk_of_chunk.index_copy_(0, chunk_id, local_chunk)
        valid_chunk = torch.zeros(J_max, dtype=torch.bool, device=device)
        valid_chunk.index_fill_(0, chunk_id, True)

        seq_lens_chunk = seq_lens[prompt_of_chunk]
        start_local = lchunk_of_chunk * C
        len_j = torch.clamp(seq_lens_chunk - start_local, max=C) * valid_chunk

        cache_key = (device, C, tuple(self.ema_scales))
        powers = self._sampled_power_cache.get(cache_key)
        if powers is None or powers.size(1) < T + 1:
            exponents = torch.arange(T + 1, device=device, dtype=torch.float32)
            powers = torch.pow(r.unsqueeze(1), exponents.unsqueeze(0))
            self._sampled_power_cache[cache_key] = powers

        within_end = sampled_chunk_end(
            feat_seq,
            prompt_of_chunk,
            start_local,
            len_j,
            offsets,
            a,
            powers,
            C,
        )

        end_of = start_local + len_j
        dist_c = start_local.unsqueeze(1) - end_of.unsqueeze(0)
        same_prompt = prompt_of_chunk.unsqueeze(1) == prompt_of_chunk.unsqueeze(0)
        prop = same_prompt & (dist_c >= 0)
        P = torch.gather(
            powers,
            1,
            dist_c.clamp(min=0).reshape(1, -1).expand(self.n_scales, -1),
        ).reshape(self.n_scales, J_max, J_max) * prop
        carry = torch.matmul(
            P, within_end.permute(1, 0, 2)
        ).permute(1, 0, 2)
        decay_init = torch.gather(
            powers,
            1,
            start_local.reshape(1, -1).expand(self.n_scales, -1),
        ).transpose(0, 1)
        carry = (
            carry + decay_init.unsqueeze(-1) * ema_old[prompt_of_chunk].to(torch.float32)
        ).contiguous()

        return sampled_output(
            feat_seq,
            sample_idx,
            prompt_idx,
            local_pos,
            chunk_id,
            start_local,
            offsets,
            carry,
            a,
            powers,
            C,
        )


# Retain a small ring per (CUDA device, dtype).  Calls may temporarily spill past
# this limit when every cached slot is in flight; those allocations are returned
# to PyTorch's pinned caching allocator, which records the H2D stream and does not
# recycle their storage until the asynchronous copy completes.
_STAGING_CACHE_LIMIT = 8
_U_STAGING: dict = {}
_U_STAGING_LOCK = threading.Lock()


class _PinnedStagingSlot:
    __slots__ = ("buffer", "copy_done", "in_flight")

    def __init__(self, n: int, dtype: torch.dtype) -> None:
        self.buffer = torch.empty(n, dtype=dtype, pin_memory=True)
        self.copy_done = torch.cuda.Event()
        self.in_flight = False

    def available(self) -> bool:
        return not self.in_flight or self.copy_done.query()


class _PinnedStagingPool:
    __slots__ = ("slots", "next_slot")

    def __init__(self) -> None:
        self.slots = []
        self.next_slot = 0

    def acquire(self, n: int, dtype: torch.dtype) -> _PinnedStagingSlot:
        count = len(self.slots)
        for offset in range(count):
            index = (self.next_slot + offset) % count
            slot = self.slots[index]
            if not slot.available():
                continue
            if slot.buffer.numel() < n:
                slot.buffer = torch.empty(n, dtype=dtype, pin_memory=True)
            self.next_slot = (index + 1) % count
            return slot

        slot = _PinnedStagingSlot(n, dtype)
        if count < _STAGING_CACHE_LIMIT:
            self.slots.append(slot)
            self.next_slot = 0 if len(self.slots) == 1 else (count + 1) % len(self.slots)
        return slot


def _staged_to_device(u_cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a CPU tensor through reusable pinned memory without blocking the host."""
    if device.type != "cuda":
        return u_cpu.to(device)
    n = u_cpu.numel()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device_index, u_cpu.dtype)
    with _U_STAGING_LOCK:
        pool = _U_STAGING.get(key)
        if pool is None:
            pool = _PinnedStagingPool()
            _U_STAGING[key] = pool
        # Event.query is non-blocking.  If every ring slot is still feeding an H2D
        # copy, acquire returns a fresh cached (or bounded one-shot spill) slot.
        slot = pool.acquire(n, u_cpu.dtype)
        view = slot.buffer[:n]
        view.copy_(u_cpu)
        result = view.to(device, non_blocking=True)
        slot.copy_done.record(torch.cuda.current_stream(device))
        slot.in_flight = True
    return result


def build_score_gather_index(
    seq_lens: torch.Tensor,
    total_tokens: int,
    prob: float,
    device: Optional[torch.device] = None,
    pad_multiple: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = seq_lens.device
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    B = seq_lens.size(0)
    mean = prob * total_tokens / B
    k_max = max(1, math.ceil(mean + _SCORE_KMAX_SIGMA * math.sqrt(mean)) + 1)
    if pad_multiple > 1:
        k_max = -(-k_max // pad_multiple) * pad_multiple
    offsets = torch.cumsum(seq_lens, dim=0) - seq_lens                       # [B]
    last_indices = offsets + seq_lens - 1                                    # [B]

    prompt_idx = torch.repeat_interleave(
        torch.arange(B, device=device), seq_lens, output_size=total_tokens
    )
    if generator is not None and generator.device.type != device.type:
        u = _staged_to_device(torch.rand(total_tokens, generator=generator), device)
    else:
        u = torch.rand(total_tokens, device=device, generator=generator)
    order = torch.argsort(prompt_idx.to(u.dtype) + u)
    cols = torch.arange(k_max, device=device)
    block_pos = (offsets.unsqueeze(1) + cols.unsqueeze(0)).clamp(max=total_tokens - 1)
    gather_idx = order[block_pos]                                            # [B, K_max]
    u_g = u[gather_idx]                                                      # [B, K_max]
    valid = (u_g < prob) & (cols.unsqueeze(0) < seq_lens.unsqueeze(1))       # [B, K_max]
    gather_idx = gather_idx.clone()
    gather_idx[:, 0] = last_indices
    valid[:, 0] = True
    return gather_idx, valid


def shard_score_columns(
    gather_idx: torch.Tensor,
    valid: torch.Tensor,
    rank: int,
    world: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert world > 0 and 0 <= rank < world, f"bad (rank, world) = ({rank}, {world})"
    return (
        gather_idx[:, rank::world].contiguous(),
        valid[:, rank::world].contiguous(),
    )
