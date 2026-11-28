# python3 -m pguard_vllm.backbones : backbone registry self-test.

from .base import BackboneRegistry, default_backbone
from . import Qwen3_5PGuardForConditionalGeneration

if __name__ == "__main__":
    # in-module self-test: python3 -m pguard_vllm.backbones
    assert BackboneRegistry.names() == ["qwen3_5"], BackboneRegistry.names()
    adapter = default_backbone()
    assert adapter.name == "qwen3_5"
    assert adapter.arch_name == "Qwen3_5PGuardForConditionalGeneration"
    cls = Qwen3_5PGuardForConditionalGeneration
    assert issubclass(cls, object) and hasattr(cls, "run_guard_step")
    assert hasattr(cls, "backbone_adapter")
    assert cls.backbone_adapter.name == "qwen3_5"
    print("backbones self-test OK (registry:", BackboneRegistry.names(), ")")
