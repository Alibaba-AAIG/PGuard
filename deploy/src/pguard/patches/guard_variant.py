import os

from ..config import (
    PGUARD_BREAKABLE_GRAPH as _ENABLED,
    PGUARD_MODELING_VARIANT,
    PGUARD_SCORE_INTERVAL as _INTERVAL,
)

_CFC_VARIANT = (PGUARD_MODELING_VARIANT or "").strip().lower() == "cfc"

if _ENABLED:
    # SGLang 0.5.11 selects the breakable backend through this environment flag.
    os.environ.setdefault("SGLANG_USE_BREAKABLE_CUDA_GRAPH", "1")

# Each captured batch-size graph records its own guard segment boundaries.
GUARD_SEGMENT_ATTR = "_pguard_guard_segment"
GUARD_END_SEGMENT_ATTR = "_pguard_guard_end_segment"
_hit = False
_decision_ready = False
_step = 0
_hook_installed = False


def enabled() -> bool:
    return _ENABLED


def get_breakable_cuda_graph_module():
    from .breakable_patch import _module

    return _module


def _capture_segment_index(capture) -> int:
    """0.5.11 registers at begin; 0.5.17 registers at end."""
    segments = capture.cuda_graph._segments
    current = getattr(capture, "_current_graph", None)
    if current is not None:
        registered = any(segment is current for segment in segments)
        return len(segments) - int(registered)
    return len(segments) - 1


def break_before_guard() -> None:
    """Start and identify the guard segment during graph capture."""
    module = get_breakable_cuda_graph_module()
    if module is None:
        return
    module.break_graph()
    capture = module._current_capture_var.get()
    if capture is None:
        return
    graph = capture.cuda_graph
    index = _capture_segment_index(capture)
    previous = getattr(graph, GUARD_SEGMENT_ATTR, None)
    if previous is not None and previous != index:
        if _CFC_VARIANT:
            # Ambiguous CFC boundaries must never permit a partial replay.
            setattr(graph, GUARD_SEGMENT_ATTR, -1)
            return
        raise RuntimeError(
            "one graph recorded multiple PGuard segments: "
            f"{previous} and {index}"
        )
    setattr(graph, GUARD_SEGMENT_ATTR, index)


def break_after_guard() -> None:
    """Mark the score boundary so replay cannot skip state or output copies."""
    module = get_breakable_cuda_graph_module()
    if module is None:
        return
    module.break_graph()
    capture = module._current_capture_var.get()
    if capture is None:
        return
    graph = capture.cuda_graph
    if getattr(graph, GUARD_SEGMENT_ATTR, None) is None:
        return
    end = _capture_segment_index(capture)
    previous = getattr(graph, GUARD_END_SEGMENT_ATTR, None)
    if previous is not None and previous != end:
        end = -1
    setattr(graph, GUARD_END_SEGMENT_ATTR, end)


def note_hook_installed() -> None:
    global _hook_installed
    _hook_installed = True


def begin_forward(is_decode: bool) -> bool:
    global _step, _hit, _decision_ready
    if _INTERVAL <= 0:
        _hit = False
    elif is_decode:
        _step += 1
        _hit = _step % _INTERVAL == 0
    else:
        _hit = True
    _decision_ready = True
    return _hit


def decide() -> bool:
    return begin_forward(True)


def pending_decision() -> bool:
    if _hook_installed:
        return _hit
    return begin_forward(True)


def record_eager_guard_execution() -> None:
    from .breakable_patch import _module

    capture = _module._current_capture_var.get() if _module is not None else None
    if capture is None and not _decision_ready:
        decide()


def consume_decision() -> bool:
    """Consume exactly one replay/eager decision, never a previous forward's value."""
    global _decision_ready
    if not _decision_ready:
        decide()
    _decision_ready = False
    return _hit


def last_decision() -> bool:
    return _hit
