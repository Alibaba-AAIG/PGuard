from __future__ import annotations
import torch
import torch.nn as nn


class RecurrentStateBuffer(nn.Module):
    def __init__(
        self,
        max_req: int,
        mem_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        assert max_req > 0, f"max_req must be >0, got {max_req}"
        assert mem_dim > 0, f"mem_dim must be >0, got {mem_dim}"
        assert num_classes >= 0, f"num_classes must be >=0, got {num_classes}"
        self.max_req = max_req
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.register_buffer(
            "gru_h", torch.zeros(max_req, mem_dim, dtype=dtype), persistent=False
        )
        if num_classes > 0:
            self.register_buffer(
                "hist_risk_max",
                torch.full((max_req, num_classes), float("-inf"), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.hist_risk_max = None

    def reset_slots(self, slot_ids: torch.Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        slot_ids = slot_ids.to(self.gru_h.device, dtype=torch.long)
        self.gru_h.index_fill_(0, slot_ids, 0.0)
        if self.hist_risk_max is not None:
            self.hist_risk_max.index_fill_(0, slot_ids, float("-inf"))

    def reset_slots_masked(self, slot_ids: torch.Tensor, fresh: torch.Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        slot_ids = slot_ids.to(self.gru_h.device, dtype=torch.long)
        keep = (~fresh).to(self.gru_h.dtype)
        self.gru_h[slot_ids] = self.gru_h[slot_ids] * keep.view(-1, 1)
        if self.hist_risk_max is not None:
            self.hist_risk_max[slot_ids] = torch.where(
                fresh.view(-1, 1),
                torch.full_like(self.hist_risk_max[slot_ids], float("-inf")),
                self.hist_risk_max[slot_ids],
            )

    def read_slots(self, slot_ids: torch.Tensor) -> torch.Tensor:
        slot_ids = slot_ids.to(self.gru_h.device, dtype=torch.long)
        return self.gru_h.index_select(0, slot_ids)

    def write_slots(self, slot_ids: torch.Tensor, h: torch.Tensor) -> None:
        slot_ids = slot_ids.to(self.gru_h.device, dtype=torch.long)
        self.gru_h.index_copy_(0, slot_ids, h.to(self.gru_h.dtype))

    def update_risk_max(
        self, slot_ids: torch.Tensor, chunk_max: torch.Tensor
    ) -> torch.Tensor:
        assert self.hist_risk_max is not None, (
            "hist_risk_max 未初始化（构造 RecurrentStateBuffer 时 num_classes=0）"
        )
        slot_ids = slot_ids.to(self.hist_risk_max.device, dtype=torch.long)
        cur = self.hist_risk_max.index_select(0, slot_ids)
        updated = torch.maximum(cur, chunk_max.to(cur.dtype))
        self.hist_risk_max.index_copy_(0, slot_ids, updated)
        return updated
