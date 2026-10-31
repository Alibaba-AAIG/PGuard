"""Qwen3.5 PGuard stateless/seq/cfc model-registration entry."""

import importlib

from ..config import PGUARD_OVERWRITE_MODEL, modeling_variant

_VARIANT_MODULES = {
    "stateless": ".variants.qwen3_5_pguard_stateless",
    "seq": ".variants.qwen3_5_pguard_seq",
    "cfc": ".variants.qwen3_5_pguard_cfc",
}

if PGUARD_OVERWRITE_MODEL:
    _variant = modeling_variant()
    if _variant not in _VARIANT_MODULES:
        raise ValueError(
            f"unknown PGUARD_MODELING_VARIANT={_variant!r} for Qwen3.5, "
            f"expect one of {sorted(_VARIANT_MODULES)}"
        )
    _impl = importlib.import_module(_VARIANT_MODULES[_variant], package=__package__)
    Qwen3_5ForConditionalGeneration = _impl.Qwen3_5ForConditionalGeneration
    Qwen3_5MoeForConditionalGeneration = _impl.Qwen3_5MoeForConditionalGeneration
    Qwen3_5PGuardForConditionalGeneration = _impl.Qwen3_5PGuardForConditionalGeneration
    Qwen3_5PGuardMoeForConditionalGeneration = _impl.Qwen3_5PGuardMoeForConditionalGeneration
    EntryClass = [
        Qwen3_5PGuardForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5PGuardMoeForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    ]
else:
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    )

    EntryClass = [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration]
