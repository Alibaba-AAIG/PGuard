import importlib

from ..config import PGUARD_OVERWRITE_MODEL, modeling_variant

_VARIANT_MODULES = {
    "stateless": ".variants.deepseek_v4_pguard_stateless",
    "seq": ".variants.deepseek_v4_pguard_seq",
    "cfc": ".variants.deepseek_v4_pguard_cfc",
}

if PGUARD_OVERWRITE_MODEL:
    _variant = modeling_variant()
    if _variant not in _VARIANT_MODULES:
        raise ValueError(
            f"unknown PGUARD_MODELING_VARIANT={_variant!r} for DeepSeek-V4, "
            f"expect one of {sorted(_VARIANT_MODULES)}"
        )
    _impl = importlib.import_module(_VARIANT_MODULES[_variant], package=__package__)
    DeepseekV4PGuardForCausalLM = _impl.DeepseekV4PGuardForCausalLM
    DeepseekV4ForCausalLM = _impl.DeepseekV4ForCausalLM
    EntryClass = [DeepseekV4PGuardForCausalLM, DeepseekV4ForCausalLM]
else:
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    EntryClass = [DeepseekV4ForCausalLM]
