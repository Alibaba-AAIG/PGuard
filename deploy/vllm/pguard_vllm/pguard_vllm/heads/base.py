# Guard-head contract and registry, modeled after vLLM's ModelRegistry.
#
# A safety head is an nn.Module with two faces:
#   * the vendored PGuard forward math (training-faithful; see common.py and
#     the per-head modules);
#   * the engine-facing `score_step` contract: one engine step's per-request
#     captured features in -> per-request flat class-probability vectors out.
#
# Heads own their recurrent state (a StatePool) and the GPU-side running
# score aggregation. They never touch the score store or the request
# lifecycle: the guard engine (model.py) slices the batch by request,
# dispatches rows to each head, and persists the returned vectors.
#
# Adding a head variant = subclass GuardHeadBase, decorate it with
# @GuardHeadRegistry.register("<pguard_version>"), ship the checkpoint --
# nothing else in the patch needs to change (the engine dispatches through
# the registry-built objects, not through per-variant branches).
#
# In-module self-test: python3 -m pguard_vllm.heads.base

import logging
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn

from ..pguard_config import get_config
from ..state_pool import StatePool

logger = logging.getLogger("pguard_vllm")

# checkpoint dtype key (PGUARD_HEAD_DTYPE) -> torch dtype
HEAD_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32, "": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def resolve_head_dtype(key: str) -> torch.dtype:
    """Normalized PGUARD_HEAD_DTYPE key -> torch dtype (unknown -> fp32)."""
    return HEAD_DTYPES.get(key, torch.float32)


@dataclass
class HeadStepResult:
    """One head's scored step, ready for the engine to persist.

    flats[i] is request req_ids[i]'s flat task-major class-probability
    vector ([task0 tok0..tokN | task1 ...], length ns[i] * total_cls) on the
    CPU -- heads do ONE device-to-host transfer per step (see
    _to_cpu_flats)."""

    req_ids: list[str]
    flats: list[list[float]]
    ns: list[int]
    is_decodes: list[bool]


def pad_rows(rows):
    """Pad ragged (C, n_i, H) row features into (B, C, max_n, H) + mask."""
    C = rows[0][1].shape[0]
    H = rows[0][1].shape[2]
    max_n = max(f.shape[1] for _, f in rows)
    device = rows[0][1].device
    padded = torch.zeros(len(rows), C, max_n, H, device=device,
                         dtype=torch.float32)
    mask = torch.zeros(len(rows), max_n, dtype=torch.bool, device=device)
    for i, (_, f) in enumerate(rows):
        padded[i, :, :f.shape[1], :] = f
        mask[i, :f.shape[1]] = True
    return padded, mask


class GuardHeadRegistry:
    """pguard_version -> head class; vLLM ModelRegistry workalike.

    Heads register at import time via the @register decorator, so the
    checkpoint's `pguard_version` field alone selects the implementation.
    Re-registering the same kind (only possible when a module is executed
    twice, e.g. `python3 -m pguard_vllm.heads.lasttok` re-runs the module
    after the package import) warns and overwrites instead of raising."""

    _heads: ClassVar[dict[str, type["GuardHeadBase"]]] = {}

    @classmethod
    def register(cls, kind: str):
        def _wrap(head_cls):
            if kind in cls._heads and cls._heads[kind] is not head_cls:
                logger.warning("pguard_vllm: pguard_version %r re-registered "
                               "with %s (was %s); keeping the newest",
                               kind, head_cls.__name__,
                               cls._heads[kind].__name__)
            cls._heads[kind] = head_cls
            head_cls.kind = kind
            return head_cls

        return _wrap

    @classmethod
    def resolve(cls, kind: str) -> type["GuardHeadBase"]:
        if kind not in cls._heads:
            raise NotImplementedError(
                f"unsupported pguard_version: {kind!r} "
                f"(registered: {cls.kinds()})")
        return cls._heads[kind]

    @classmethod
    def kinds(cls) -> list[str]:
        return sorted(cls._heads)


class GuardHeadBase(nn.Module):
    """Base class for all PGuard safety heads."""

    kind: ClassVar[str] = ""

    def __init__(self, config):
        super().__init__()
        self._head_cfg = config
        self._layer_pos: list[int] = []
        self._task_layout: list[tuple[str, int, int]] = []
        self._total_cls = 0
        self._dtype = torch.float32
        self._pool: StatePool | None = None

    # -- engine wiring -----------------------------------------------------
    def bind(self, *, device, dtype, layer_pos, total_cls, task_layout,
             max_slots) -> None:
        """Attach engine context, called once after checkpoint load: which
        positions of the union capture stack this head reads, the flat-score
        layout, and the compute device/dtype. Allocates the state pool."""
        self._layer_pos = list(layer_pos)
        self._total_cls = int(total_cls)
        self._task_layout = list(task_layout)
        self.to(device=device, dtype=dtype)
        self._dtype = dtype
        self._pool = self._build_pool(device, max_slots)

    def _build_pool(self, device, max_slots: int) -> StatePool:
        """Stateless default: score-aggregation buffers only."""
        return StatePool(self.kind, device, max_slots=max_slots,
                         total_cls=self._total_cls)

    # -- scoring contract --------------------------------------------------
    def score_step(self, prefill_rows, decode_rows,
                   score: bool = True) -> HeadStepResult | None:
        """Score one engine step for this head.

        prefill_rows / decode_rows: lists of (req_id, feats) with feats
        (C_union, n, H) fp32 from the capture stack (n=1 for decode rows).
        score=False thins the decode-side SCORING only (cadence); recurrent
        state must still advance. Returns the scored vectors, or None when
        this step scores nothing for this head."""
        raise NotImplementedError

    # -- lifecycle delegates (engine bookkeeping goes through these) -------
    def contains(self, req_id: str) -> bool:
        return self._pool is not None and self._pool.contains(req_id)

    def tracked_req_ids(self) -> list[str]:
        return self._pool.req_ids() if self._pool is not None else []

    def free_req(self, req_id: str) -> None:
        if self._pool is not None:
            self._pool.free_req(req_id)

    def evict_oldest_beyond(self, limit: int) -> None:
        """FIFO safety valve when a pool grows past the tracked-state cap."""
        if self._pool is None:
            return
        while len(self._pool.slot_of) > limit:
            self._pool.free_req(self._pool.req_ids()[0])

    def finalize(self, req_id: str) -> tuple | None:
        """(has_p, has_d, plast, dlast, dmax) CPU vectors for a finished
        request, releasing its slot. None when this head never pooled it."""
        if self._pool is None or not self._pool.contains(req_id):
            return None
        out = self._pool.acc_read(self._pool.slot_of[req_id])
        self._pool.free_req(req_id)
        return out

    # -- shared scoring helpers --------------------------------------------
    @property
    def fuse_ops(self) -> bool:
        return get_config().fuse_ops

    def _acc_update(self, req_ids: list[str], flats, ns, is_decodes) -> None:
        """Fold each request's newest scored-token vector into its slot's GPU
        accumulator (one small kernel chain per scoring step); read back once
        on finish (finalize) so the final record is O(1)."""
        pool = self._pool
        if pool is None or pool.acc_last_p is None or not flats:
            return
        device = flats[0].device
        slot_t = pool.slots_for(req_ids, device)
        last_vecs = []
        for f, n in zip(flats, ns):
            parts = []
            for _task, start, stop in self._task_layout:
                n_cls = stop - start
                # task i's block sits at [n*start, n*stop); its last token is
                # the block's tail
                parts.append(f[n * start + (n - 1) * n_cls:n * stop])
            last_vecs.append(torch.cat(parts))
        pool.acc_update(slot_t, torch.stack(last_vecs), is_decodes)

    def _to_cpu_flats(self, probs_per_req, req_ids: list[str],
                      is_decodes) -> list[list[float]]:
        """Move a per-request stack of probability tensors to the host with
        ONE device-to-host transfer, as flat task-major vectors
        ([task0 tok0..tokN | task1 ...], length n * total_cls per request;
        layout mirrors the official interrupt_policy column ranges). Also
        folds the newest scored-token vectors into the GPU accumulators."""
        flats, ns = [], []
        for probs in probs_per_req:
            flats.append(torch.cat([p.reshape(-1) for p in probs]))
            ns.append(int(probs[0].shape[0]) if probs else 0)
        if not flats:
            return []
        self._acc_update(req_ids, flats, ns, is_decodes)
        flat = torch.cat(flats).tolist()  # single D2H transfer
        out, ofs = [], 0
        for n in ns:
            out.append(flat[ofs:ofs + n * self._total_cls])
            ofs += n * self._total_cls
        return out


if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.heads.base
    from types import SimpleNamespace

    assert GuardHeadRegistry.kinds() == [], "import side effects leaked"

    @GuardHeadRegistry.register("dummy")
    class _Dummy(GuardHeadBase):
        def score_step(self, prefill_rows, decode_rows, score=True):
            return None

    assert GuardHeadRegistry.resolve("dummy") is _Dummy
    assert _Dummy.kind == "dummy"

    # re-registration warns and overwrites (runpy double-execution safety)
    @GuardHeadRegistry.register("dummy")
    class _Dummy2(GuardHeadBase):
        def score_step(self, prefill_rows, decode_rows, score=True):
            return None

    assert GuardHeadRegistry.resolve("dummy") is _Dummy2
    GuardHeadRegistry._heads.pop("dummy")
    try:
        GuardHeadRegistry.resolve("dummy")
        raise AssertionError("unregistered kind must raise")
    except NotImplementedError:
        pass

    # resolve_head_dtype covers the shipped keys and falls back to fp32
    for key, want in (("fp32", torch.float32), ("bf16", torch.bfloat16),
                      ("fp16", torch.float16), ("", torch.float32),
                      ("junk", torch.float32)):
        assert resolve_head_dtype(key) is want, key

    # finalize/evict delegate to the pool
    head = _Dummy(SimpleNamespace())
    head.bind(device="cpu", dtype=torch.float32, layer_pos=[0],
              total_cls=3, task_layout=[("t", 0, 3)], max_slots=2)
    assert not head.contains("r1") and head.tracked_req_ids() == []
    assert head.finalize("r1") is None and head.evict_oldest_beyond(1) is None
    assert head.fuse_ops is get_config().fuse_ops
    GuardHeadRegistry._heads.pop("dummy", None)
    print("heads.base self-test OK (registry, dtype map, lifecycle)")
