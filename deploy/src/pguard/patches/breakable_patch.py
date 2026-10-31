"""Replay all graph segments except an explicitly marked, unrequested guard."""

import importlib
import logging
import os

import torch


from . import guard_variant

logger = logging.getLogger(__name__)
_patched = False
_warned_unmarked = False
_MODULE_CANDIDATES = (
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph",
    "sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph",
)


def _resolve_breakable_module():
    """Resolve the SGLang 0.5.17 or 0.5.11 breakable graph module."""
    for name in _MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if not (exc.name == name or name.startswith(exc.name + ".")):
                raise
    return None


_module = _resolve_breakable_module()


def break_graph() -> None:
    """Record the closing break used by existing seq/stateless call sites."""
    if _module is None:
        return
    capture = _module._current_capture_var.get()
    graph = capture.cuda_graph if capture is not None else None
    if (
        graph is not None
        and getattr(graph, guard_variant.GUARD_SEGMENT_ATTR, None) is not None
        and getattr(graph, guard_variant.GUARD_END_SEGMENT_ATTR, None) is None
    ):
        guard_variant.break_after_guard()
    else:
        _module.break_graph()


def _segment_indices_to_replay(
    count: int, guard_index, hit: bool, guard_end=None, *, require_end: bool = False
) -> list[int]:
    """Require closed CFC boundaries while preserving start-only graph layouts."""
    all_indices = list(range(count))
    if guard_index is None:
        return all_indices
    if type(guard_index) is not int or not 0 <= guard_index < count:
        if require_end or guard_end is not None:
            return all_indices
        raise RuntimeError(
            f"invalid recorded PGuard segment {guard_index!r} for {count} segments"
        )
    if require_end or guard_end is not None:
        if (
            type(guard_end) is not int
            or not 0 < guard_index < guard_end < count
            or guard_end != guard_index + 1
        ):
            return all_indices
    return all_indices if hit else [
        index for index in range(count) if index != guard_index
    ]


def patch_breakable_replay() -> None:
    global _patched
    if _patched or not guard_variant.enabled():
        return
    if _module is None:
        logger.warning("No breakable CUDA graph implementation; guard runs every step")
        return

    def _conditional_replay(self):
        global _warned_unmarked
        hit = guard_variant.pending_decision()
        segments = self._segments
        guard_index = getattr(self, guard_variant.GUARD_SEGMENT_ATTR, None)
        guard_end = getattr(self, guard_variant.GUARD_END_SEGMENT_ATTR, None)
        replay_indices = _segment_indices_to_replay(
            len(segments), guard_index, hit, guard_end,
            require_end=guard_variant._CFC_VARIANT,
        )
        if (
            not hit and len(segments) >= 2
            and len(replay_indices) == len(segments) and not _warned_unmarked
        ):
            _warned_unmarked = True
            logger.warning(
                "[breakable] missing/invalid PGuard boundary (start=%r, end=%r); "
                "replaying all %d segments",
                guard_index, guard_end, len(segments),
            )
        skip_index = guard_index if len(replay_indices) < len(segments) else None


        stream = torch.cuda.current_stream()
        token = _module._current_stream_var.set(stream)
        try:
            for index, segment in enumerate(segments):
                if index != skip_index:
                    segment.replay()
                # Break callbacks maintain BCG output and copy semantics. They must run
                # even for the skipped guard segment, matching upstream replay behavior.
                if index < len(self._break_fns):
                    self._break_fns[index]()
        finally:
            _module._current_stream_var.reset(token)

    _module.BreakableCUDAGraph.replay = _conditional_replay
    _patched = True
    logger.info("[PATCH] conditional PGuard replay enabled (pid=%d)", os.getpid())
