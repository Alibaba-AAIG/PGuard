# DeepSeek-V4 backbone adapter.
#
# Architecture layout (vLLM vllm/models/deepseek_v4/, present since
# v0.31.0; per-platform implementations under nvidia/ and amd/):
#   DeepseekV4ForCausalLM
#     └── model: DeepseekV4Model
#           └── layers: decoder layer list (start_layer..end_layer)
# No EAGLE3-style aux capture API: hooks + eager mode applies.
#
# Real-machine status: boot-level verification pending (tiny checkpoint).

from typing import Callable

import torch.nn as nn

from .base import BackboneAdapter, BackboneRegistry, make_pguard_class


class DeepseekV4Adapter(BackboneAdapter):

    name = "deepseek_v4"
    arch_name = "DeepseekV4PGuardForCausalLM"

    def base_cls(self) -> type:
        from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
        return DeepseekV4ForCausalLM

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def root_module(self, model: nn.Module) -> nn.Module:
        return model.model

    def enable_aux(self, model: nn.Module, hf_indices: tuple) -> bool:
        return False


_backbone = DeepseekV4Adapter()
_backbone.module = __name__
BackboneRegistry.register(_backbone)

# concrete architecture class registered by plugin.py
DeepseekV4PGuardForCausalLM = make_pguard_class(_backbone)
