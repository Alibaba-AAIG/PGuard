# Minimal shared dependencies of the vendored PGuard heads, adapted from
# Alibaba-AAIG/PGuard train/src (Apache-2.0):
#   train/src/utils/model_utils.py (BasicClassificationHead, StableDropout, ...)
#   train/src/utils/modeling_outputs.py (PGuardMultiTasksOutputForCls)
# plus the inference-only fused paths shared by all head variants
# (mirrors the official deploy package's PGUARD_FUSE_ADAPTER /
# PGUARD_FUSE_CLASSIFIERS; mathematically identical, they only cut kernel
# launches and nn.Module dispatch; enabled per call site via config.fuse_ops).
#
# This module holds NO engine logic: pure vendored building blocks.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput


class _DictConfig:
    """Attribute-access view over a flat dict; supports `in` for key checks."""

    def __init__(self, d):
        self.__dict__.update(d)

    def __contains__(self, key):
        return key in self.__dict__


def config_from_dict(d):
    return _DictConfig(d)


# --------------------------------------------------------------------------
# output container (train/src/utils/modeling_outputs.py)
# --------------------------------------------------------------------------
@dataclass
class PGuardMultiTasksOutputForCls(ModelOutput):
    classification_logits: torch.FloatTensor = None
    full_classification_logits: torch.FloatTensor = None
    attention_mask: torch.LongTensor = None
    hidden_states: tuple = None
    gate_topk_scores: torch.FloatTensor = None
    gate_topk_indices: torch.LongTensor = None
    gate_all_scores: torch.FloatTensor = None
    loss: torch.Tensor = None
    loss_dict: dict = None
    input_ids: torch.Tensor = None
    token_type_ids: torch.Tensor = None
    fusion_attentions: torch.FloatTensor = None


# --------------------------------------------------------------------------
# dropout / classification head (train/src/utils/model_utils.py)
# --------------------------------------------------------------------------
class DropoutContext:
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def _get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask
    return mask, dropout


class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = _get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        return grad_output, None


class StableDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self._get_context())
        return x

    def _get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        return self.drop_prob


class BasicClassificationHead(nn.Module):
    """Token-level classification head."""

    def __init__(self, config):
        super().__init__()
        dropout_type = getattr(config, "dropout_type", "stable") or "stable"
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if dropout_type == "stable":
            self.dropout = StableDropout(config.dropout_prob)
        else:
            self.dropout = nn.Dropout(config.dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, feature, **kwargs):
        feature = self.dense(feature)
        feature = self.act_fn(feature)
        feature = self.dropout(feature)
        return self.out_proj(feature)


# --------------------------------------------------------------------------
# shared adapter block (identical in the lasttok and ema heads)
# --------------------------------------------------------------------------
class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        return self.output(self.intermediate(x), x)


# --------------------------------------------------------------------------
# inference-only fused paths (eval only; identical math, fewer launches)
# --------------------------------------------------------------------------
def fused_adapter_forward(block, x):
    """AdapterBlock (BertIntermediate+BertOutput) in 4 ops instead of ~6:
    addmm -> gelu -> addmm -> LayerNorm(residual + proj). Eval only."""
    x2d = x.reshape(-1, x.shape[-1])
    inter = torch.addmm(block.intermediate.dense.bias, x2d,
                        block.intermediate.dense.weight.t())
    inter = F.gelu(inter)
    proj = torch.addmm(block.output.dense.bias, inter,
                       block.output.dense.weight.t())
    out = block.output.LayerNorm(proj + x2d)
    return out.reshape(x.shape)


def classifiers_forward(classifiers, h):
    """Apply a ModuleList[BasicClassificationHead] sharing one trunk.

    The per-head dense+gelu is one concatenated addmm+gelu; the small
    out_proj layers run as a single grouped matmul over zero-padded weights.
    h: (..., hidden) -> list of (..., n_cls) logits, one per head.
    """
    mods = list(classifiers)
    key = (h.device, h.dtype,
           tuple(m.dense.weight.data_ptr() for m in mods),
           tuple(m.out_proj.weight.data_ptr() for m in mods))
    cache = getattr(classifiers, "_pguard_fused_cache", None)
    if cache is None or cache[0] != key:
        hidden = mods[0].dense.weight.shape[1]
        w1 = torch.cat([m.dense.weight for m in mods], dim=0)
        b1 = torch.cat([m.dense.bias for m in mods])
        cmax = max(m.out_proj.weight.shape[0] for m in mods)
        w2 = torch.zeros(len(mods), cmax, hidden, device=h.device, dtype=h.dtype)
        b2 = torch.zeros(len(mods), cmax, device=h.device, dtype=h.dtype)
        for i, m in enumerate(mods):
            n = m.out_proj.weight.shape[0]
            w2[i, :n] = m.out_proj.weight
            b2[i, :n] = m.out_proj.bias
        cache = (key, w1, b1, w2, b2)
        classifiers._pguard_fused_cache = cache
    _, w1, b1, w2, b2 = cache
    lead = h.shape[:-1]
    x = h.reshape(-1, h.shape[-1])
    inter = F.gelu(torch.addmm(b1, x, w1.t()))
    n, s, hidden = inter.shape[0], len(mods), w1.shape[1]
    inter = inter.view(n, s, hidden).transpose(0, 1)        # (S, N, hidden)
    out = torch.baddbmm(b2.unsqueeze(1), inter, w2.transpose(1, 2))
    res = []
    for i, m in enumerate(mods):
        c = m.out_proj.weight.shape[0]
        res.append(out[i, :, :c].reshape(*lead, c))
    return res
