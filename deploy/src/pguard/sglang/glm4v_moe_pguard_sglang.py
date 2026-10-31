import importlib

from ..config import PGUARD_OVERWRITE_MODEL, modeling_variant

_VARIANT_MODULES = {
    "stateless": ".variants.glm4v_moe_pguard_stateless",
    "seq": ".variants.glm4v_moe_pguard_seq",
    "cfc": ".variants.glm4v_moe_pguard_cfc",
}

if PGUARD_OVERWRITE_MODEL:
    _variant = modeling_variant()
    if _variant not in _VARIANT_MODULES:
        raise ValueError(
            f"unknown PGUARD_MODELING_VARIANT={_variant!r} for GLM-4.5V (glm4v_moe), "
            f"expect one of {sorted(_VARIANT_MODULES)}"
        )
    _impl = importlib.import_module(_VARIANT_MODULES[_variant], package=__package__)
    Glm4vMoePGuardForConditionalGeneration = _impl.Glm4vMoePGuardForConditionalGeneration
    Glm4vMoeForConditionalGeneration = _impl.Glm4vMoeForConditionalGeneration
    EntryClass = [
        Glm4vMoePGuardForConditionalGeneration,
        Glm4vMoeForConditionalGeneration,
    ]
else:
    from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration

    EntryClass = [Glm4vMoeForConditionalGeneration]
