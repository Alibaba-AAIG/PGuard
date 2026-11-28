# Backbone adapters: registry package.
#
# Adapter modules self-register into BackboneRegistry on import -- the same
# pattern as the heads/ package. Each adapter contributes one architecture
# class (built by make_pguard_class) that plugin.py registers with vLLM's
# ModelRegistry; the engine picks one via --hf-overrides.
#
# Backbones tracked for adaptation (vLLM upstream support required before an
# adapter can be written; official PGuard supports them on SGLang):
#   DeepSeek-V4  -- awaits vLLM upstream model implementation
#   GLM-4V-MoE   -- awaits vLLM upstream model implementation
#
# In-module self-test: python3 -m pguard_vllm.backbones

import logging

from .base import BackboneAdapter, BackboneRegistry, make_pguard_class
from .qwen3_5 import Qwen3_5Adapter, Qwen3_5PGuardForConditionalGeneration
from .deepseek_v4 import DeepseekV4Adapter, DeepseekV4PGuardForCausalLM
from .glm4v_moe import Glm4vMoeAdapter, Glm4vMoePGuardForConditionalGeneration

logger = logging.getLogger("pguard_vllm")

__all__ = [
    "BackboneAdapter", "BackboneRegistry", "make_pguard_class",
    "Qwen3_5Adapter", "Qwen3_5PGuardForConditionalGeneration",
    "DeepseekV4Adapter", "DeepseekV4PGuardForCausalLM",
    "Glm4vMoeAdapter", "Glm4vMoePGuardForConditionalGeneration",
    "default_backbone",
]

from .base import default_backbone  # noqa: E402  (after registry populated)
