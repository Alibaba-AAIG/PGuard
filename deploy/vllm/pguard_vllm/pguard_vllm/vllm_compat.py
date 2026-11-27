# vLLM version-compat layer.
#
# Single place that knows how internals differ across vLLM versions:
# GPUModelRunner classes, SchedulerOutput fields, aux hidden-state storage.
# patch_runner.py (the actual monkey-patching) reads everything from here.
#
# Version-specific knowledge lives in CompatProfile objects held in a
# registry (vLLM ModelRegistry style): each profile declares the runner
# candidates and aux storage spots for a range of versions; `profile_for()`
# picks the longest version-prefix match, falling back to the generic
# profile. Adding a new vLLM version: run
# `python3 -m pguard_vllm.vllm_compat` in that environment, register/extend
# a profile until the report is green, then run the acceptance flow in
# tests/README.md.
#
# In-module self-test / report: python3 -m pguard_vllm.vllm_compat

from dataclasses import dataclass

from .pguard_config import get_config

# GPUModelRunner candidates, most-compatible first. The legacy class is the
# default worker in 0.25.1; the V2 class (vllm.v1.worker.gpu.model_runner)
# takes over when VLLM_USE_V2_MODEL_RUNNER=1.
_BASE_RUNNER_CANDIDATES = (
    "vllm.v1.worker.gpu_model_runner:GPUModelRunner",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner",
)

# aux hidden-state storage spots, probed in order per step (attribute paths
# from the runner object).
_AUX_SPOTS_GENERIC = (
    ("execute_model_state", "aux_hidden_states"),
    ("cudagraph_manager", "aux_hidden_states"),
)


@dataclass(frozen=True)
class CompatProfile:
    """Patch assumptions for a range of vLLM versions."""

    name: str
    version_prefixes: tuple[str, ...]          # "" matches everything
    runner_candidates: tuple[str, ...]
    aux_spots: tuple[tuple[str, ...], ...]
    notes: str = ""


_PROFILES: dict[str, CompatProfile] = {}


def register_profile(profile: CompatProfile) -> CompatProfile:
    _PROFILES[profile.name] = profile
    return profile


GENERIC = register_profile(CompatProfile(
    name="generic",
    version_prefixes=("",),
    runner_candidates=_BASE_RUNNER_CANDIDATES,
    aux_spots=_AUX_SPOTS_GENERIC,
    notes="legacy runner default; V2 optional",
))


V0_25 = register_profile(CompatProfile(
    name="v0.25.x",
    version_prefixes=("0.25.",),
    runner_candidates=_BASE_RUNNER_CANDIDATES,
    aux_spots=_AUX_SPOTS_GENERIC,
    notes="verified baseline",
))


def profile_for(version: str) -> CompatProfile:
    """Longest registered version-prefix match; GENERIC as fallback."""
    best = GENERIC
    best_len = 0
    for p in _PROFILES.values():
        for prefix in p.version_prefixes:
            if prefix and version.startswith(prefix) and len(prefix) > best_len:
                best, best_len = p, len(prefix)
    return best


def current_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception as e:
        return "import failed: %r" % (e,)


def runner_class_paths(profile: CompatProfile | None = None) -> list[str]:
    profile = profile or profile_for(current_version())
    candidates = list(profile.runner_candidates)
    for item in get_config().runner_candidates:
        if item not in candidates:
            candidates.append(item)
    return candidates


def resolve_runner_classes(profile: CompatProfile | None = None):
    """Import every candidate: ([(path, cls)], [(path, error)])."""
    resolved, failed = [], []
    for path in runner_class_paths(profile):
        mod, _, name = path.partition(":")
        try:
            cls = getattr(__import__(mod, fromlist=[name]), name)
            resolved.append((path, cls))
        except Exception as e:
            failed.append((path, repr(e)))
    return resolved, failed


# -- SchedulerOutput -----------------------------------------------------
# Fields used (0.25.1 / 0.34.0): num_scheduled_tokens (dict req_id -> n, in
# batch row order), scheduled_new_reqs (req_id / prompt_token_ids /
# num_computed_tokens), scheduled_cached_reqs (req_ids + num_computed_tokens),
# finished_req_ids.

_prompt_lens: dict[str, int | None] = {}  # req_id -> prompt token length


def scheduled_token_counts(scheduler_output) -> dict | None:
    """req_id -> scheduled token count for this step (batch order), or None."""
    try:
        nst = scheduler_output.num_scheduled_tokens
        return dict(nst) if nst else None
    except Exception:
        return None


def decode_flags(scheduler_output, counts: dict) -> list[bool] | None:
    """Per-row True=decode, derived without GPU syncs: a request is decoding
    once num_computed_tokens >= prompt_len (the runner's own is_prefilling
    predicate). None when any row is unknown (caller falls back)."""
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
        for rid in counts:
            plen = _prompt_lens.get(rid)
            c = computed.get(rid)
            if plen is not None and c is not None:
                flags.append(c >= plen)
            else:
                flags.append(None)  # unknown -> model fallback decides
        if any(f is None for f in flags):
            return None
        return flags
    except Exception:
        return None


def finished_ids(scheduler_output) -> list:
    return list(getattr(scheduler_output, "finished_req_ids", None) or [])


def forget_prompt_lens(req_ids) -> None:
    for rid in req_ids:
        _prompt_lens.pop(rid, None)


# -- aux hidden states -----------------------------------------------------

def resolve_aux(runner, stash, profile: CompatProfile | None = None):
    """Aux hidden states for the step that just ran: the _model_forward
    interceptor's stash first, then the profile's per-runner storage spots
    (first non-None wins)."""
    if stash is not None:
        return stash
    if runner is None:
        return None
    profile = profile or profile_for(current_version())
    for path in profile.aux_spots:
        obj = runner
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj:
            return obj
    return None


# -- report / self-check ----------------------------------------------------

def _field_names(cls) -> set[str]:
    """Field names of a dataclass or msgspec struct (version-dependent)."""
    if hasattr(cls, "__struct_fields__"):          # msgspec
        return set(cls.__struct_fields__)
    if hasattr(cls, "__dataclass_fields__"):
        return set(cls.__dataclass_fields__)
    return set(getattr(cls, "__annotations__", {}))


def _new_request_data_cls():
    for name in ("NewRequestData", "ScheduledNewRequestData"):
        try:
            from vllm.v1.core.sched import output as sched_output
            cls = getattr(sched_output, name, None)
            if cls is not None:
                return cls
        except Exception:
            pass
    return None


def report() -> dict:
    """Everything the patch depends on, resolved against the installed vLLM."""
    version = current_version()
    profile = profile_for(version)
    resolved, failed = resolve_runner_classes(profile)
    out: dict = {
        "vllm_version": version,
        "profile": profile.name,
        "runner_classes": [p for p, _ in resolved],
        "runner_failures": failed,
        "scheduler_fields": {},
    }
    try:
        from vllm.v1.core.sched.output import SchedulerOutput
        fields = _field_names(SchedulerOutput)
        for f in ("num_scheduled_tokens", "scheduled_new_reqs",
                  "scheduled_cached_reqs", "finished_req_ids"):
            out["scheduler_fields"][f] = f in fields
        new_req = _new_request_data_cls()
        if new_req is not None:
            nfields = _field_names(new_req)
            for f in ("req_id", "prompt_token_ids", "num_computed_tokens"):
                out["scheduler_fields"]["new_req." + f] = f in nfields
    except Exception as e:
        out["scheduler_fields"]["error"] = repr(e)
    return out


def self_check() -> int:
    """0 = the installed vLLM exposes everything the patch needs."""
    rep = report()
    print("vLLM version:      ", rep["vllm_version"])
    print("compat profile:    ", rep["profile"])
    print("runner classes:    ", rep["runner_classes"] or "NONE")
    for path, err in rep["runner_failures"]:
        print("  unavailable:     ", path, err)
    missing = [f for f, ok in rep["scheduler_fields"].items() if not ok]
    print("scheduler fields:  ",
          "all present" if not missing else "MISSING %s" % missing)
    ok = bool(rep["runner_classes"]) and not missing
    print("result:            ", "OK - patchable" if ok
          else "FAIL - extend candidates (PGUARD_RUNNER_CANDIDATES) or "
               "vllm_compat.py")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(self_check())
