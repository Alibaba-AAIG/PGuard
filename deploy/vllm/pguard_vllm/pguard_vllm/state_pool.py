# Slot-pooled per-request recurrent state for the guard heads.
#
# Replaces per-request dicts of tensors with preallocated GPU buffers indexed
# by small integer slots (the official deploy package's RecurrentStateBuffer
# pattern): one index_select / index_put_ per step instead of per-request dict
# lookups, device casts and fresh stack allocations. Memory is fixed; slots
# are zeroed and freed as soon as the scheduler reports the request finished,
# so a recycled slot never leaks a previous request's state (all guard work
# runs on the default stream, so reuse is stream-ordered).
#
# One pool per head name. EMA pools carry (C, n_scales, H) vectors per
# request; GRU pools carry (mem_dim,) hidden states; stateless heads need no
# pool. Buffer size is bounded by PGUARD_MAX_SLOTS (default 64 >= max_num_seqs
# 32); exhaustion evicts the oldest request FIFO (same safety valve as the
# old dict logic).
#
# In-module self-test: python3 -m pguard_vllm.state_pool

import os
from collections import deque

import torch

_MAX_SLOTS = int(os.environ.get("PGUARD_MAX_SLOTS", "64"))


class StatePool:
    """Fixed-capacity slot pool with optional EMA and GRU state buffers."""

    def __init__(self, name: str, device, ema_shape=None, mem_dim: int = 0,
                 max_slots: int | None = None, total_cls: int = 0):
        self.name = name
        self.max_slots = max_slots or _MAX_SLOTS
        self.slot_of: dict[str, int] = {}
        self._free = deque(range(self.max_slots))
        self._order: list[str] = []  # FIFO eviction order when exhausted
        self.ema = (torch.zeros(self.max_slots, *ema_shape, device=device,
                                dtype=torch.float32)
                    if ema_shape is not None else None)
        self.gru = (torch.zeros(self.max_slots, mem_dim, device=device,
                                dtype=torch.float32)
                    if mem_dim else None)
        # GPU-side running score aggregation (R2): updated with one small
        # kernel per scoring step, read back once when the request finishes
        # (final record) so pop() is O(1) instead of O(steps).
        self.total_cls = total_cls
        self.acc_last_p = (torch.zeros(self.max_slots, total_cls,
                                       device=device, dtype=torch.float32)
                           if total_cls else None)
        self.acc_last_d = (torch.zeros(self.max_slots, total_cls,
                                       device=device, dtype=torch.float32)
                           if total_cls else None)
        self.acc_max_d = (torch.zeros(self.max_slots, total_cls,
                                      device=device, dtype=torch.float32)
                          if total_cls else None)
        self.acc_has_p = (torch.zeros(self.max_slots, dtype=torch.bool,
                                      device=device) if total_cls else None)
        self.acc_has_d = (torch.zeros(self.max_slots, dtype=torch.bool,
                                      device=device) if total_cls else None)

    # -- slot bookkeeping --------------------------------------------------
    def slots_for(self, req_ids, device) -> torch.Tensor:
        """Slot index tensor for the given requests, allocating on demand."""
        for rid in req_ids:
            if rid not in self.slot_of:
                self._alloc(rid)
        return torch.tensor([self.slot_of[rid] for rid in req_ids],
                            device=device, dtype=torch.long)

    def _alloc(self, req_id: str) -> int:
        if not self._free:
            victim = self._order.pop(0)
            self._release_slot(victim)
        slot = self._free.popleft()
        self.slot_of[req_id] = slot
        self._order.append(req_id)
        return slot

    def _release_slot(self, req_id: str) -> None:
        slot = self.slot_of.pop(req_id, None)
        if slot is None:
            return
        for buf in (self.ema, self.gru, self.acc_last_p, self.acc_last_d,
                    self.acc_max_d):
            if buf is not None:
                buf[slot].zero_()
        if self.acc_has_p is not None:
            self.acc_has_p[slot] = False
            self.acc_has_d[slot] = False
        self._free.append(slot)

    def free_req(self, req_id: str) -> None:
        """Release a finished request's slot (zeroes its state)."""
        if req_id in self.slot_of:
            self._order.remove(req_id)
            self._release_slot(req_id)

    def contains(self, req_id: str) -> bool:
        return req_id in self.slot_of

    def req_ids(self) -> list[str]:
        return list(self.slot_of)

    # -- state access ------------------------------------------------------
    def ema_read(self, slot_t: torch.Tensor) -> torch.Tensor:
        return self.ema[slot_t]              # (B, C, n_scales, H) copy

    def ema_write(self, slot_t: torch.Tensor, states: torch.Tensor) -> None:
        self.ema[slot_t] = states

    def gru_read(self, slot_t: torch.Tensor) -> torch.Tensor:
        return self.gru[slot_t]              # (B, mem_dim) copy

    def gru_write(self, slot_t: torch.Tensor, h: torch.Tensor) -> None:
        self.gru[slot_t] = h

    # -- score aggregation (R2) --------------------------------------------
    def acc_update(self, slot_t: torch.Tensor, last_vecs: torch.Tensor,
                   is_decodes) -> None:
        """Update running aggregation with each request's newest scored-token
        vector (B, total_cls). `is_decodes` is a bool/list of bools (one per
        row; mixed prefill/decode batches are split by mask). One small
        kernel chain per scoring step."""
        if isinstance(is_decodes, (list, tuple)):
            dec = torch.tensor([bool(x) for x in is_decodes],
                               device=last_vecs.device, dtype=torch.bool)
        else:
            dec = torch.full((last_vecs.shape[0],), bool(is_decodes),
                             device=last_vecs.device, dtype=torch.bool)
        pre = ~dec
        if pre.any():
            idx = slot_t[pre]
            self.acc_last_p[idx] = last_vecs[pre]
            self.acc_has_p[idx] = True
        if dec.any():
            idx = slot_t[dec]
            self.acc_last_d[idx] = last_vecs[dec]
            self.acc_max_d[idx] = torch.maximum(self.acc_max_d[idx],
                                                last_vecs[dec])
            self.acc_has_d[idx] = True

    def acc_read(self, slot: int):
        """(has_prefill, has_decode, plast, dlast, dmax) CPU lists for one
        slot — called once when the request finishes."""
        out = []
        for buf in (self.acc_has_p, self.acc_has_d, self.acc_last_p,
                    self.acc_last_d, self.acc_max_d):
            out.append(buf[slot].tolist())
        hp, hd, plast, dlast, dmax = out
        return (bool(hp), bool(hd), plast, dlast, dmax)


if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.state_pool
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool = StatePool("ema", dev, ema_shape=(3, 4, 8), mem_dim=0, max_slots=2,
                     total_cls=5)

    s = pool.slots_for(["r1", "r2"], dev)
    assert s.tolist() == [0, 1]
    pool.ema_write(s, torch.arange(2 * 3 * 4 * 8, dtype=torch.float32,
                                   device=dev).reshape(2, 3, 4, 8))
    assert pool.ema_read(s)[0, 0, 0, 0].item() == 0.0

    # r3 evicts r1 (FIFO); r1's slot must be zeroed before reuse
    s2 = pool.slots_for(["r3", "r2"], dev)
    assert pool.slot_of["r3"] == 0 and pool.slot_of["r2"] == 1
    assert pool.ema[0].abs().sum().item() == 0.0
    assert pool.contains("r2") and not pool.contains("r1")

    pool.free_req("r2")
    assert not pool.contains("r2") and pool.ema[1].abs().sum().item() == 0.0
    s3 = pool.slots_for(["r4"], dev)
    assert pool.slot_of["r4"] == 1

    # aggregation: decode updates max/last; prefill updates last only
    v1 = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], device=dev)
    v2 = torch.tensor([[0.05, 0.9, 0.05, 0.05, 0.05]], device=dev)
    st = pool.slots_for(["r5"], dev)
    pool.acc_update(st, v1, [True])
    pool.acc_update(st, v2, [True])
    hp, hd, plast, dlast, dmax = pool.acc_read(pool.slot_of["r5"])
    assert hd and not hp
    assert abs(dlast[1] - 0.9) < 1e-6 and abs(dmax[0] - 0.1) < 1e-6
    pool.acc_update(st, v1, [False])
    hp, hd, plast, dlast, dmax = pool.acc_read(pool.slot_of["r5"])
    assert hp and abs(plast[0] - 0.1) < 1e-6
    print("state_pool self-test OK (alloc/evict/zero/reuse/aggregate)")
