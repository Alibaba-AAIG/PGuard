# PGuard safety heads: registry package.
#
# Head modules self-register their class (keyed by the checkpoint's
# `pguard_version` field) into GuardHeadRegistry on import -- the same
# pattern vLLM's model registry uses. build_head_from_checkpoint resolves
# the class from the registry, so a new head variant ships as one new module
# in this package and nothing else in the patch changes.
#
# In-module self-test: python3 -m pguard_vllm.heads

import json
import logging
import os
from types import SimpleNamespace

import torch

from ..pguard_config import get_config
from .base import (GuardHeadBase, GuardHeadRegistry, HeadStepResult,
                   pad_rows, resolve_head_dtype, HEAD_DTYPES)
from .lasttok import PluginGuard
from .ema import PluginGuardEMA
from .gru import PluginGuardGRU

logger = logging.getLogger("pguard_vllm")

__all__ = [
    "GuardHeadBase", "GuardHeadRegistry", "HeadStepResult", "pad_rows",
    "resolve_head_dtype", "HEAD_DTYPES", "PluginGuard", "PluginGuardEMA",
    "PluginGuardGRU", "load_head_config", "build_head_from_checkpoint",
]


def load_head_config(head_path: str) -> SimpleNamespace:
    """Head checkpoint config.json -> attribute namespace."""
    with open(os.path.join(head_path, "config.json")) as f:
        raw = json.load(f)
    # PluginGuard accesses config.classifier[...] as a dict and the rest as
    # attributes, mirroring the HF config layout used during training.
    ns = SimpleNamespace(**raw)
    # runtime override for the GRU head's prefill summarization (pool | ema)
    cfc_mode = get_config().cfc_prefill_mode
    if cfc_mode and getattr(ns, "pguard_version", "") == "gru":
        ns.prefill_mode = cfc_mode
    return ns


def build_head_from_checkpoint(head_path: str):
    """Build the safety head from a PGuard checkpoint directory (on CPU).

    The class comes from GuardHeadRegistry via the config's pguard_version;
    unknown versions fail loudly here (startup fail-fast)."""
    cfg = load_head_config(head_path)
    version = cfg.pguard_version
    model = GuardHeadRegistry.resolve(version)(cfg)
    state = torch.load(os.path.join(head_path, "pytorch_model.bin"),
                       map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("pguard_vllm: loaded head %s from %s", version, head_path)
    return cfg, model

