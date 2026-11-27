# Runtime patch for vLLM GPUModelRunner (PGuard scoring hookup).
#
# Two patches, both fail-safe (never break generation):
#
# 1. execute_model wrapper -- real scheduler steps only (CUDA-graph capture
#    and warmups go through _dummy_run, which calls the model directly and
#    bypasses this patch). It derives per-row decode flags straight from the
#    SchedulerOutput (num_computed_tokens >= prompt_len, the same predicate
#    the runner uses for is_prefilling), dispatches finished_req_ids, and
#    calls model.run_guard_step() AFTER the engine step returns. Scoring here
#    keeps the guard out of graph capture and off the sampling critical path.
#
# 2. _model_forward wrapper -- when PGUARD_CAPTURE_MODE=aux, the model returns
#    (hidden_states, aux_hidden_states); the runner unpacks it when its
#    use_aux_hidden_state_outputs flag is set (set here at construction), and
#    this wrapper stashes the aux list so run_guard_step can score from it
#    (fresh data on every eager forward AND CUDA-graph replay: the graph
#    manager copies graph outputs into persistent buffers inside the graph).
#
# 0.25.1 has TWO independent GPU runner classes and the worker uses the LEGACY
# one by default (vllm.v1.worker.gpu_model_runner.GPUModelRunner); the new one
# (vllm.v1.worker.gpu.model_runner.GPUModelRunner) is used when
# use_v2_model_runner=True. Both are patched; each wrapper must bind its own
# original (factory function - closure late-binding would otherwise make both
# wrappers call the same original).
#
# Version compat (R-compat): runner classes are resolved from a candidate
# list (extend with PGUARD_RUNNER_CANDIDATES="mod:Class,...") and the aux
# hidden states are probed across the known storage spots per step
# (_resolve_aux), so newer/older runner generations only need candidate
# entries -- not a rewrite. Everything fails safe and leaves traces.

import logging
import os

from .pguard_config import get_config

logger = logging.getLogger("pguard_vllm")

_current_sched = None         # req_id -> scheduled token count (this step)
_step_flags = None            # per-row decode flags for this step
_last_aux = None              # aux hidden states from the last forward
_prompt_lens = {}             # req_id -> prompt token length
_patched = False
_DBG = {"n": 0}

# Runner class candidates, most-compatible first. Extra candidates for other
# vLLM versions come via PGUARD_RUNNER_CANDIDATES
# ("vllm.v1.worker.gpu_model_runner.GPUModelRunner,vllm.v1.worker.gpu.model_runner.GPUModelRunnerV2,...").
_BASE_RUNNER_CANDIDATES = (
    "vllm.v1.worker.gpu_model_runner:GPUModelRunner",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner",
)


def _runner_candidates():
    candidates = list(_BASE_RUNNER_CANDIDATES)
    for item in get_config().runner_candidates:
        if item not in candidates:
            candidates.append(item)
    return candidates


def _resolve_aux(runner):
    """Aux hidden states for the step that just ran, probing the known
    storage spots across runner generations (first non-None wins):
    1. the _model_forward interceptor's stash (legacy runner)
    2. runner.execute_model_state.aux_hidden_states (legacy + V2)
    3. runner.cudagraph_manager.aux_hidden_states (V2 persistent buffers)"""
    if _last_aux is not None:
        return _last_aux
    if runner is not None:
        st = getattr(runner, "execute_model_state", None)
        aux = getattr(st, "aux_hidden_states", None)
        if aux:
            return aux
        gm = getattr(runner, "cudagraph_manager", None)
        aux = getattr(gm, "aux_hidden_states", None)
        if aux:
            return aux
    return None

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


def _compute_decode_flags(scheduler_output, num_scheduled_tokens):
    """Per-row True=decode, derived from the SchedulerOutput without GPU syncs
    or file IO: a request is decoding once num_computed_tokens >= prompt_len
    (the runner's own is_prefilling predicate)."""
    try:
        computed = {}
        for nd in scheduler_output.scheduled_new_reqs:
            plen = (len(nd.prompt_token_ids)
                    if nd.prompt_token_ids is not None else None)
            _prompt_lens[nd.req_id] = plen
            computed[nd.req_id] = nd.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        ids = getattr(cached, "req_ids", None) or []
        comps = getattr(cached, "num_computed_tokens", None) or []
        for rid, c in zip(ids, comps):
            computed[rid] = c
        flags = []
        for rid in num_scheduled_tokens:
            plen = _prompt_lens.get(rid)
            c = computed.get(rid)
            if plen is not None and c is not None:
                flags.append(c >= plen)
            else:
                flags.append(None)  # unknown -> model fallback decides
        if any(f is None for f in flags):
            return None
        return flags
    except Exception as e:
        if _DBG["n"] < 6:
            _DBG["n"] += 1
            _trace("decode-flag derivation failed: %r" % (e,))
        return None


def _note_finished(req_ids):
    for rid in req_ids:
        _prompt_lens.pop(rid, None)
    try:
        from .model import _MODEL
        if _MODEL is not None:
            _MODEL.note_finished(req_ids)
    except Exception:
        pass


def _score_after(sched, flags, aux, runner=None):
    if not sched:
        return
    # probe the known aux storage spots across runner generations
    aux = _resolve_aux(runner) if aux is None else aux
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
        global _current_sched, _step_flags, _last_aux
        _last_aux = None
        sched = None
        flags = None
        if scheduler_output is not None:
            try:
                nst = scheduler_output.num_scheduled_tokens
                sched = dict(nst) if nst else None
            except Exception as e:
                if _DBG["n"] < 6:
                    _DBG["n"] += 1
                    _trace("sched capture failed: %r" % (e,))
                sched = None
        if sched:
            flags = _compute_decode_flags(scheduler_output, sched)
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                _note_finished(finished)
        _current_sched = sched
        _step_flags = flags
        try:
            result = orig_execute_model(self, scheduler_output, *args, **kwargs)
        finally:
            _current_sched = None
            _step_flags = None
        # score after the engine step, outside any forward context; aux (if
        # any) was stashed by the _model_forward wrapper during the step, or
        # is read back from the runner's execute_model_state (V2 runner)
        _score_after(sched, flags, _last_aux, self)
        _last_aux = None
        return result

    execute_model._pguard_patched = True
    return execute_model


def _make_model_forward_wrapper(orig_model_forward):
    def _model_forward(self, *args, **kwargs):
        out = orig_model_forward(self, *args, **kwargs)
        global _last_aux
        if type(out) is tuple and len(out) == 2 and isinstance(out[1], list):
            _last_aux = out[1]
        else:
            _last_aux = None
        return out

    _model_forward._pguard_patched = True
    return _model_forward


def _install_patch():
    global _patched
    if _patched:
        return

    targets = []
    for mod, name in (item.split(":", 1) for item in _runner_candidates()):
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
            if cls not in targets:
                targets.append(cls)
                _trace("runner class imported: %s.%s" % (mod, name))
        except Exception as e:
            _trace("runner import failed: %s: %r" % (mod, e))

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
                    "(aux_mode=%s)", get_config().capture_mode == "aux")
    else:
        # a guard service that silently runs without scoring is a security
        # failure -- refuse to start instead
        raise RuntimeError(
            "pguard_vllm: could not patch any GPUModelRunner candidate "
            "(checked: %s); fix PGUARD_RUNNER_CANDIDATES or the vLLM version"
            % ", ".join(_runner_candidates()))
