# Runtime patch for vLLM GPUModelRunner (PGuard scoring hookup).
#
# Two patches, both fail-safe (never break generation):
#
# 1. execute_model wrapper -- real scheduler steps only (CUDA-graph capture
#    and warmups go through _dummy_run, which calls the model directly and
#    bypasses this patch). It derives per-row decode flags from the
#    SchedulerOutput, dispatches finished_req_ids, and calls
#    model.run_guard_step() AFTER the engine step returns. Scoring here keeps
#    the guard out of graph capture and off the sampling critical path.
#
# 2. _model_forward wrapper -- when PGUARD_CAPTURE_MODE=aux, the model returns
#    (hidden_states, aux_hidden_states); the runner unpacks it when its
#    use_aux_hidden_state_outputs flag is set (set here at construction), and
#    this wrapper stashes the aux list so run_guard_step can score from it
#    (fresh data on every eager forward AND CUDA-graph replay: the graph
#    manager copies graph outputs into persistent buffers inside the graph).
#
# 0.25.1 has TWO independent GPU runner classes and the worker uses the LEGACY
# one by default; the V2 one is used when use_v2_model_runner=True. Both are
# patched; each wrapper must bind its own original (factory function --
# closure late-binding would otherwise make both wrappers call the same
# original).
#
# All version-dependent knowledge (runner classes, SchedulerOutput fields,
# aux storage spots) lives in vllm_compat.py.

import logging
import os

from .pguard_config import get_config
from .vllm_compat import (current_version, decode_flags, finished_ids,
                          forget_prompt_lens, profile_for, resolve_aux,
                          resolve_runner_classes, runner_class_paths,
                          scheduled_token_counts)

logger = logging.getLogger("pguard_vllm")

_last_aux = None              # aux hidden states from the last forward
_patched = False
_PROFILE = None               # CompatProfile resolved at patch time
_DBG = {"n": 0}

_TRACE = None  # resolved lazily from the config (PGUARD_TRACE)


def _trace(msg):
    """Ground-truth probe: direct file write, independent of logging setup."""
    try:
        path = get_config().trace_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, ("%d %s\n" % (os.getpid(), msg)).encode())
        finally:
            os.close(fd)
    except Exception:
        pass


def _note_finished(req_ids):
    forget_prompt_lens(req_ids)
    try:
        from .model import _MODEL
        if _MODEL is not None:
            _MODEL.note_finished(req_ids)
    except Exception:
        pass


def _score_after(sched, flags, aux, runner=None):
    if not sched:
        return
    aux = resolve_aux(runner, aux, _PROFILE)
    try:
        from .model import _MODEL
        model = _MODEL
    except Exception:
        return
    if model is None or not getattr(model, "_pguard_ready", False):
        return
    try:
        model.run_guard_step(sched, flags, aux)
    except Exception:
        logger.warning("pguard_vllm: scoring failed", exc_info=True)


def _make_wrapper(orig_execute_model):
    def execute_model(self, scheduler_output, *args, **kwargs):
        global _last_aux
        _last_aux = None
        sched = None
        flags = None
        if scheduler_output is not None:
            sched = scheduled_token_counts(scheduler_output)
            if sched is None and _DBG["n"] < 6:
                _DBG["n"] += 1
                _trace("sched capture failed")
            if sched:
                flags = decode_flags(scheduler_output, sched)
                if flags is None and _DBG["n"] < 6:
                    _DBG["n"] += 1
                    _trace("decode-flag derivation failed")
                finished = finished_ids(scheduler_output)
                if finished:
                    _note_finished(finished)
        result = orig_execute_model(self, scheduler_output, *args, **kwargs)
        # score after the engine step, outside any forward context; aux (if
        # any) was stashed by the _model_forward wrapper during the step, or
        # is read back from the runner (see vllm_compat.resolve_aux)
        _score_after(sched, flags, _last_aux, self)
        _last_aux = None
        return result

    execute_model._pguard_patched = True
    return execute_model


def _make_model_forward_wrapper(orig_model_forward):
    def _model_forward(self, *args, **kwargs):
        global _last_aux
        out = orig_model_forward(self, *args, **kwargs)
        if type(out) is tuple and len(out) == 2 and isinstance(out[1], list):
            _last_aux = out[1]
        else:
            _last_aux = None
        return out

    _model_forward._pguard_patched = True
    return _model_forward


def _install_patch():
    global _patched, _PROFILE
    if _patched:
        return

    _PROFILE = profile_for(current_version())
    resolved, failed = resolve_runner_classes(_PROFILE)
    for path, err in failed:
        _trace("runner import failed: %s: %s" % (path, err))
    targets = [cls for _, cls in resolved]
    _trace("compat profile: %s" % _PROFILE.name)

    aux_mode = get_config().capture_mode == "aux"
    patched_any = False
    for cls in targets:
        try:
            if getattr(cls.execute_model, "_pguard_patched", False):
                continue
            if aux_mode and not getattr(cls, "_pguard_aux_init", False):
                # The aux tuple output must be enabled for graph capture and
                # replay, but V2's load_model asserts aux requires a drafter
                # (speculative_config is None here). Keep the flag False
                # through load_model (skipping that branch entirely -- aux
                # layers are already set by the model itself) and flip it
                # right after, before CUDA-graph capture begins.
                orig_init = cls.__init__

                def init_with_aux(instance, *a, **k):
                    orig_init(instance, *a, **k)
                    orig_load_model = instance.load_model

                    def load_model_then_aux(*a2, **k2):
                        res = orig_load_model(*a2, **k2)
                        instance.use_aux_hidden_state_outputs = True
                        return res

                    instance.load_model = load_model_then_aux

                cls.__init__ = init_with_aux
                cls._pguard_aux_init = True
            if hasattr(cls, "_model_forward") and \
                    not getattr(cls._model_forward, "_pguard_patched", False):
                cls._model_forward = _make_model_forward_wrapper(
                    cls._model_forward)
            cls.execute_model = _make_wrapper(cls.execute_model)
            patched_any = True
            _trace("patch installed on %s.%s" % (cls.__module__, cls.__name__))
        except Exception as e:
            # one runner class failing must not abort the other
            _trace("patch failed on %s: %r" % (getattr(cls, "__name__", cls), e))

    if patched_any:
        _patched = True
        logger.info("pguard_vllm: GPUModelRunner patched for guard capture "
                    "(aux_mode=%s)", aux_mode)
    else:
        # a guard service that silently runs without scoring is a security
        # failure -- refuse to start instead
        raise RuntimeError(
            "pguard_vllm: could not patch any GPUModelRunner candidate "
            "(checked: %s); fix PGUARD_RUNNER_CANDIDATES or the vLLM version"
            % ", ".join(runner_class_paths(_PROFILE)))
