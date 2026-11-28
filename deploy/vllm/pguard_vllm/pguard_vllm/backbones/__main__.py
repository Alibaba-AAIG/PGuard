# python3 -m pguard_vllm.backbones : backbone registry self-test.

from .base import BackboneRegistry, default_backbone
from . import (
    Qwen3_5PGuardForConditionalGeneration,
    DeepseekV4PGuardForCausalLM,
    Glm4vMoePGuardForConditionalGeneration,
)

if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.backbones
    assert BackboneRegistry.names() == ["deepseek_v4", "glm4v_moe", "qwen3_5"], \
        BackboneRegistry.names()
    adapter = default_backbone()
    assert adapter.name == "qwen3_5"
    assert adapter.arch_name == "Qwen3_5PGuardForConditionalGeneration"
    for cls, name in (
            (Qwen3_5PGuardForConditionalGeneration, "qwen3_5"),
            (DeepseekV4PGuardForCausalLM, "deepseek_v4"),
            (Glm4vMoePGuardForConditionalGeneration, "glm4v_moe")):
        assert hasattr(cls, "run_guard_step"), name
        assert hasattr(cls, "backbone_adapter"), name
        assert cls.backbone_adapter.name == name
    print("backbones self-test OK (registry:", BackboneRegistry.names(), ")")
