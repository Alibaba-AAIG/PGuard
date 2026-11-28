# Backbone adapter contract and registry, modeled after vLLM's
# ModelRegistry.
#
# A backbone adapter answers the three questions the guard mixin needs from
# any backbone architecture: where the decoder layers live (hook mount
# points), which module resolves device/parameters, and how to enable the
# aux hidden-state capture (when the architecture supports it). Combined
# with make_pguard_class it produces the concrete architecture class that
# plugin.py registers under vllm.general_plugins.
#
# Adding a backbone = one adapter module in this package: subclass
# BackboneAdapter, register it with @BackboneRegistry.register, call
# make_pguard_class(adapter) and export the generated class. The engine
# picks the architecture via --hf-overrides.

from typing import Callable

import torch.nn as nn

from ..pguard_config import get_config


class BackboneRegistry:
    """backbone name -> adapter; vLLM ModelRegistry workalike."""

    _adapters: dict[str, "BackboneAdapter"] = {}

    @classmethod
    def register(cls, adapter: "BackboneAdapter") -> "BackboneAdapter":
        if adapter.name in cls._adapters:
            raise ValueError(f"backbone {adapter.name!r} already registered")
        cls._adapters[adapter.name] = adapter
        return adapter

    @classmethod
    def resolve(cls, name: str) -> "BackboneAdapter":
        if name not in cls._adapters:
            raise NotImplementedError(
                f"unknown backbone: {name!r} (registered: {cls.names()})")
        return cls._adapters[name]

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def all(cls) -> list["BackboneAdapter"]:
        return [cls._adapters[n] for n in cls.names()]


class BackboneAdapter:
    """Per-backbone access points for the guard mixin."""

    name: str = ""
    # architecture name registered into vLLM's ModelRegistry
    arch_name: str = ""
    # adapter module path; the generated architecture class lives here
    module: str = ""

    def base_cls(self) -> type:
        """The vLLM backbone model class the pguard class inherits."""
        raise NotImplementedError

    def decoder_layers(self, model: nn.Module) -> nn.ModuleList:
        """Decoder layer list; hook mount points for hooks capture mode."""
        raise NotImplementedError

    def root_module(self, model: nn.Module) -> nn.Module:
        """Module used for device/parameter resolution."""
        raise NotImplementedError

    def enable_aux(self, model: nn.Module, hf_indices: tuple) -> bool:
        """Enable EAGLE3-style aux capture for the given HF layer indices.
        Returns False when the backbone lacks the machinery (hooks+eager
        fallback applies)."""
        return False


def make_pguard_class(adapter: BackboneAdapter) -> type:
    """Construct the concrete pguard architecture class for a backbone.

    Kept annotation-free callers-safe: the mixin lives in model.py and is
    imported lazily to avoid a circular import (model.py imports this
    package's base module)."""
    from ..model import PGuardMixin

    base = adapter.base_cls()
    ns = {"backbone_adapter": adapter}
    return type(adapter.arch_name, (PGuardMixin, base), ns)


def default_backbone() -> "BackboneAdapter":
    """Backbone selected via PGUARD_BACKBONE (default: qwen3_5)."""
    return BackboneRegistry.resolve(get_config().backbone)


def register_backbone(
    name: str, arch_name: str
) -> Callable[[type[BackboneAdapter]], BackboneAdapter]:
    """Class-decorator variant: instantiates and registers the adapter."""

    def _wrap(adapter_cls):
        adapter = adapter_cls()
        adapter.name = name
        adapter.arch_name = arch_name
        BackboneRegistry.register(adapter)
        return adapter

    return _wrap
