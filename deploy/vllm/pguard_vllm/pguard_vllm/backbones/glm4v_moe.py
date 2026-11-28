# GLM-4V-MoE backbone adapter (multimodal: vision tower + text decoder).
#
# Architecture layout (vLLM vllm/model_executor/models/glm4_1v.py, present
# since v0.31.0):
#   Glm4vMoeForConditionalGeneration
#     ├── visual: Glm4vVisionTransformer
#     └── language_model: (registry-resolved text model)
#           └── model: text decoder
#                 └── layers: decoder layer list
# No EAGLE3-style aux capture API: hooks + eager mode applies.
#
# Real-machine status: boot-level verification pending (tiny checkpoint).

import torch.nn as nn

from .base import BackboneAdapter, BackboneRegistry, make_pguard_class


class Glm4vMoeAdapter(BackboneAdapter):

    name = "glm4v_moe"
    arch_name = "Glm4vMoePGuardForConditionalGeneration"

    def base_cls(self) -> type:
        from vllm.model_executor.models.glm4_1v import (
            Glm4vMoeForConditionalGeneration)
        return Glm4vMoeForConditionalGeneration

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.language_model.model.layers

    def root_module(self, model: nn.Module) -> nn.Module:
        return model.language_model.model

    def enable_aux(self, model: nn.Module, hf_indices: tuple) -> bool:
        return False


_backbone = Glm4vMoeAdapter()
_backbone.module = __name__
BackboneRegistry.register(_backbone)

# concrete architecture class registered by plugin.py
Glm4vMoePGuardForConditionalGeneration = make_pguard_class(_backbone)
