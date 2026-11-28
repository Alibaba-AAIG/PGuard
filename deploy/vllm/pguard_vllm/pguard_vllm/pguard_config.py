# Central PGuard patch configuration: every PGUARD_* environment variable is
# parsed in exactly one place, into a frozen PGuardConfig.
#
# Modules must not read os.environ directly. One lookup point keeps the knob
# inventory auditable (grep PGUARD_ hits this file plus the docker/.env docs)
# and gives the startup path a single spot to validate before the engine
# spends minutes on weight loading.
#
# Parsing semantics are exactly the ones the patch shipped with (defaults,
# strip/lower coercion, invalid-value fallbacks) so existing deployments keep
# behaving identically. int()/float() on garbage values still raise -- a loud
# startup crash beats a silently mis-configured guard.
#
# In-module self-test: python3 -m pguard_vllm.pguard_config

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PGuardConfig:
    # heads / checkpoints (raw env values; store.resolve_head_dirs applies
    # the naming scheme and precedence)
    head_path: str | None
    head_subdir: str | None
    heads: str | None
    head_dtype: str                  # normalized key for heads.HEAD_DTYPES
    # capture
    capture_layers: str              # '' = no env override (checkpoint wins)
    capture_mode: str                # 'hooks' | 'aux'
    # scoring
    score_interval: int              # 1 = every decode step, N = every Nth, 0 = never
    prefill_risk_mode: str           # 'all' | 'last'
    # runtime knobs
    max_slots: int
    fuse_ops: bool
    enforce_eager: bool
    cfc_prefill_mode: str | None     # GRU head prefill summarization override
    mm_image_limit: int              # max images per prompt (multimodal)
    # API layer
    threshold: float
    verbose: bool
    debug: bool
    interrupt: bool
    # engine patch / diagnostics
    backbone: str                    # registered backbone name (PGUARD_BACKBONE)
    runner_candidates: tuple[str, ...]   # EXTRA runner candidates (module:Class,...)
    trace_guard: bool
    trace_path: str
    store_dir: str


def _resolve_score_interval(prob_raw: str, interval_raw: str) -> int:
    """PGUARD_COPY_LOGITS_PROB / PGUARD_SCORE_INTERVAL -> steps per score.

    A direct interval wins; otherwise prob >= 1 -> 1 (every step), <= 0 -> 0
    (never score decode; state updates still run), else round(1/prob)
    (0.1 -> every 10th step, the official cadence)."""
    if interval_raw:
        return max(0, int(interval_raw))
    try:
        prob = float(prob_raw)
    except ValueError:
        prob = 1.0
    if prob <= 0.0:
        return 0
    if prob >= 1.0:
        return 1
    return max(1, round(1.0 / prob))


def _parse_runner_candidates(raw: str) -> tuple[str, ...]:
    """'mod:Class, junk, a.b:C1:C2' -> ('mod:Class', 'a.b:C1:C2')."""
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item and ":" in item:
            out.append(item)
    return tuple(out)


def _parse_config() -> PGuardConfig:
    capture_mode = os.environ.get("PGUARD_CAPTURE_MODE", "hooks").strip().lower()
    if capture_mode not in ("hooks", "aux"):
        capture_mode = "hooks"
    prefill_mode = os.environ.get("PGUARD_PREFILL_RISK_MODE", "all").strip().lower()
    if prefill_mode not in ("all", "last"):
        prefill_mode = "all"
    cfc_mode = os.environ.get("PGUARD_CFC_PREFILL_MODE", "").strip() or None
    return PGuardConfig(
        head_path=os.environ.get("PGUARD_HEAD_PATH") or None,
        head_subdir=os.environ.get("PGUARD_HEAD_SUBDIR") or None,
        heads=os.environ.get("PGUARD_HEADS") or None,
        head_dtype=os.environ.get("PGUARD_HEAD_DTYPE", "").strip().lower(),
        capture_layers=os.environ.get("PGUARD_CAPTURE_LAYERS", "").strip(),
        capture_mode=capture_mode,
        score_interval=_resolve_score_interval(
            os.environ.get("PGUARD_COPY_LOGITS_PROB", "1.0").strip(),
            os.environ.get("PGUARD_SCORE_INTERVAL", "").strip()),
        prefill_risk_mode=prefill_mode,
        max_slots=int(os.environ.get("PGUARD_MAX_SLOTS", "64")),
        fuse_ops=os.environ.get("PGUARD_FUSE_OPS", "1") != "0",
        enforce_eager=os.environ.get("PGUARD_ENFORCE_EAGER", "1") == "1",
        cfc_prefill_mode=cfc_mode,
        mm_image_limit=max(1, int(os.environ.get("PGUARD_MM_LIMIT", "4"))),
        threshold=float(os.environ.get("PGUARD_THRESHOLD", "0.5")),
        verbose=bool(int(os.environ.get("PGUARD_VERBOSE", "0"))),
        debug=bool(int(os.environ.get("PGUARD_DEBUG", "0"))),
        interrupt=bool(int(os.environ.get("PGUARD_INTERRUPT", "0"))),
        backbone=os.environ.get("PGUARD_BACKBONE", "qwen3_5").strip(),
        runner_candidates=_parse_runner_candidates(
            os.environ.get("PGUARD_RUNNER_CANDIDATES", "")),
        trace_guard=os.environ.get("PGUARD_TRACE_GUARD", "") == "1",
        trace_path=os.environ.get("PGUARD_TRACE",
                                  "/dev/shm/pguard_store/_patch_trace.log"),
        store_dir=os.environ.get("PGUARD_STORE_DIR", "/dev/shm/pguard_store"),
    )


_CONFIG = None


def get_config() -> PGuardConfig:
    """Process-local singleton, parsed from the environment on first use."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _parse_config()
    return _CONFIG


def reload_config() -> PGuardConfig:
    """Re-parse from the current environment (self-tests / late env setup)."""
    global _CONFIG
    _CONFIG = _parse_config()
    return _CONFIG


if __name__ == "__main__":
    # in-module self-test: parsing semantics must match the shipped behavior
    _CASES = [
        # (env overrides, config field, expected)
        ({}, "capture_mode", "hooks"),
        ({"PGUARD_CAPTURE_MODE": " aux "}, "capture_mode", "aux"),
        ({"PGUARD_CAPTURE_MODE": "bogus"}, "capture_mode", "hooks"),
        ({}, "score_interval", 1),
        ({"PGUARD_COPY_LOGITS_PROB": "0.1"}, "score_interval", 10),
        ({"PGUARD_COPY_LOGITS_PROB": "0.5"}, "score_interval", 2),
        ({"PGUARD_COPY_LOGITS_PROB": "1.5"}, "score_interval", 1),
        ({"PGUARD_COPY_LOGITS_PROB": "0"}, "score_interval", 0),
        ({"PGUARD_COPY_LOGITS_PROB": "junk"}, "score_interval", 1),
        ({"PGUARD_COPY_LOGITS_PROB": "0.1", "PGUARD_SCORE_INTERVAL": "3"},
         "score_interval", 3),                       # direct interval wins
        ({"PGUARD_SCORE_INTERVAL": "0"}, "score_interval", 0),
        ({}, "prefill_risk_mode", "all"),
        ({"PGUARD_PREFILL_RISK_MODE": " LAST "}, "prefill_risk_mode", "last"),
        ({"PGUARD_PREFILL_RISK_MODE": "bogus"}, "prefill_risk_mode", "all"),
        ({"PGUARD_HEAD_DTYPE": " BF16 "}, "head_dtype", "bf16"),
        ({}, "head_dtype", ""),
        ({"PGUARD_MAX_SLOTS": "32"}, "max_slots", 32),
        ({}, "max_slots", 64),
        ({"PGUARD_FUSE_OPS": "0"}, "fuse_ops", False),
        ({}, "fuse_ops", True),
        ({"PGUARD_ENFORCE_EAGER": "0"}, "enforce_eager", False),
        ({}, "enforce_eager", True),
        ({"PGUARD_RUNNER_CANDIDATES": "a.mod:Cls, junk, b:C1:C2"},
         "runner_candidates", ("a.mod:Cls", "b:C1:C2")),
        ({}, "runner_candidates", ()),
        ({"PGUARD_TRACE_GUARD": "1"}, "trace_guard", True),
        ({}, "trace_guard", False),
        ({"PGUARD_STORE_DIR": "/tmp/x"}, "store_dir", "/tmp/x"),
        ({}, "store_dir", "/dev/shm/pguard_store"),
        ({"PGUARD_THRESHOLD": "0.7"}, "threshold", 0.7),
        ({}, "threshold", 0.5),
        ({"PGUARD_INTERRUPT": "1"}, "interrupt", True),
        ({}, "interrupt", False),
        ({"PGUARD_MM_LIMIT": "8"}, "mm_image_limit", 8),
        ({}, "mm_image_limit", 4),
        ({"PGUARD_MM_LIMIT": "0"}, "mm_image_limit", 1),   # clamped to >= 1
        ({"PGUARD_BACKBONE": " deepseek_v4 "}, "backbone", "deepseek_v4"),
        ({}, "backbone", "qwen3_5"),
    ]
    bad = []
    _saved = {k: v for k, v in os.environ.items() if k.startswith("PGUARD_")}
    _touched = set(_saved)
    try:
        for overrides, field, expected in _CASES:
            for k in _touched:          # drop everything earlier cases set
                os.environ.pop(k, None)
            _touched.update(overrides)
            os.environ.update(overrides)
            got = getattr(reload_config(), field)
            if got != expected:
                bad.append("%s=%r (want %r) for %s"
                           % (field, got, expected, overrides))
    finally:
        for k in _touched:
            os.environ.pop(k, None)
        os.environ.update(_saved)
        reload_config()
    for b in bad:
        print("FAIL", b)
    print("pguard_config self-test:",
          "FAILED (%d)" % len(bad) if bad else "ok")
    raise SystemExit(1 if bad else 0)
