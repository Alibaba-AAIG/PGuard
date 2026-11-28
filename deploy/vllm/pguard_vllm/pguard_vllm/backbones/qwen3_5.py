# Qwen3.5 backbone adapter (verified baseline).
#
# Architecture layout (vLLM vllm/model_executor/models/qwen3_5.py):
#   Qwen3_5ForConditionalGeneration
#     └── language_model: Qwen3_5ForCausalLM
#           └── model: Qwen3_5Model
#                 └── layers: decoder layer list
# Aux capture uses the EAGLE3-style set_aux_hidden_state_layers on the
# language model.

from typing import Callable

import torch.nn as nn

from .base import BackboneAdapter, BackboneRegistry, make_pguard_class


class Qwen3_5Adapter(BackboneAdapter):

    name = "qwen3_5"
    arch_name = "Qwen3_5PGuardForConditionalGeneration"

    def base_cls(self) -> type:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration)
        return Qwen3_5ForConditionalGeneration

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.language_model.model.layers

    def root_module(self, model: nn.Module) -> nn.Module:
        return model.language_model.model

    def enable_aux(self, model: nn.Module, hf_indices: tuple) -> bool:
        if not hasattr(model.language_model, "set_aux_hidden_state_layers"):
            return False
        model.language_model.set_aux_hidden_state_layers(hf_indices)
        return True


_backbone = Qwen3_5Adapter()
_backbone.module = __name__
BackboneRegistry.register(_backbone)

# concrete architecture class registered by plugin.py
Qwen3_5PGuardForConditionalGeneration = make_pguard_class(_backbone)
