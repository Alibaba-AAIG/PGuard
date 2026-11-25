# Vendored PGuard safety head (Lasttok / "PluginGuard") and its minimal
# dependencies, adapted from Alibaba-AAIG/PGuard train/src (Apache-2.0):
#   train/src/model/modeling_pguard_lasttok.py
#   train/src/utils/model_utils.py (BasicClassificationHead, StableDropout, ...)
#   train/src/utils/modeling_outputs.py (PGuardMultiTasksOutputForCls)
# Only inference-relevant code is kept; training losses are stripped.

import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput

logger = logging.getLogger("pguard_vllm")

NORMAL_LABEL = "normal"


# --------------------------------------------------------------------------
# minimal config / output helpers (from train/src/utils)
# --------------------------------------------------------------------------
class _DictConfig:
    """Attribute-access view over a flat dict; supports `in` for key checks."""

    def __init__(self, d):
        self.__dict__.update(d)

    def __contains__(self, key):
        return key in self.__dict__


def config_from_dict(d):
    return _DictConfig(d)


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
# dropout / classification head (from train/src/utils/model_utils.py)
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
# inference-only fused paths (mirrors the official deploy package's
# PGUARD_FUSE_ADAPTER / PGUARD_FUSE_CLASSIFIERS); mathematically identical,
# they only cut kernel launches and nn.Module dispatch. Enabled by default,
# PGUARD_FUSE_OPS=0 falls back to the unfused module calls.
# --------------------------------------------------------------------------
_FUSE_OPS = os.environ.get("PGUARD_FUSE_OPS", "1") != "0"


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


# --------------------------------------------------------------------------
# Lasttok head ("PluginGuard", pguard_version=lasttok)
# from train/src/model/modeling_pguard_lasttok.py (inference path only)
# --------------------------------------------------------------------------
class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        return self.output(self.intermediate(x), x)


class PluginGuard(nn.Module):
    """Consumes stacked multi-layer hidden states (bs, layer, len, hidden)."""

    def __init__(self, config):
        super().__init__()
        self.top_L = 3
        self.gate = nn.Linear(config.hidden_size, 1)
        self.gate_temperature = 1.0

        bert_layer_config = BertConfig()
        bert_layer_config.add_cross_attention = False

        self.input_proj = nn.Linear(config.hidden_size, bert_layer_config.hidden_size)
        self.adapter = nn.ModuleList(
            [AdapterBlock(bert_layer_config) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=bert_layer_config.layer_norm_eps)

        self.classifiers = nn.ModuleList()
        for name, num_classes in config.classifier["classification_heads"]:
            classifier_config = config_from_dict({
                "hidden_size": bert_layer_config.hidden_size,
                "dropout_prob": config.classifier["dropout_prob"],
                "num_classes": num_classes,
                "dropout_type": config.classifier["dropout_type"],
            })
            self.classifiers.add_module(name, BasicClassificationHead(classifier_config))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_gate(self, features):
        # features: (bs, layer, sent_len, hidden_size)
        B, L, S, H = features.shape
        scores = self.gate(features).squeeze(-1)  # (B, L, S)
        scores_no_noise = scores
        if self.training:  # inference only; kept for fidelity
            U = torch.rand_like(scores)
            scores += -torch.log(-torch.log(U + 1e-8) + 1e-8)
        scores /= self.gate_temperature
        if not self.training and _FUSE_OPS and self.top_L >= L:
            # all captured layers are kept: skip topk+gather, the weighted
            # sum over the softmax weights is identical
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, S, 1)
            final_feature = (features * weights).sum(dim=1)   # (B, S, H)
            all_scores = F.softmax(scores_no_noise, dim=1)
            return final_feature, scores, None, all_scores
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)
        expanded = topk_indices.unsqueeze(-1).expand(B, self.top_L, S, H)
        topk_features = torch.gather(features, dim=1, index=expanded)
        weights = F.softmax(topk_scores, dim=1).unsqueeze(-1)
        final_feature = (topk_features * weights).sum(dim=1)  # (B, S, H)
        all_scores = F.softmax(scores_no_noise, dim=1)
        return final_feature, topk_scores, topk_indices, all_scores

    def forward(self, features, token_type_ids=None, label_type=None,
                label_marker=None, **kwargs):
        features = self.norm(features)
        features, gate_topk_scores, gate_topk_indices, gate_all_scores = \
            self.forward_gate(features)
        features = self.input_proj(features)
        if not self.training and _FUSE_OPS:
            for layer in self.adapter:
                features = fused_adapter_forward(layer, features)
            full_logits_list = classifiers_forward(self.classifiers, features)
        else:
            for layer in self.adapter:
                features = layer(features)

            full_logits_list = []
            for classifier in self.classifiers:
                full_logits_list.append(classifier(features))  # (bs, sent_len, n_cls)

        attention_mask = None
        if token_type_ids is not None:
            attention_mask = (
                torch.stack(list(token_type_ids.values()), 0).sum(0) > 0
            ).to(torch.long)

        return PGuardMultiTasksOutputForCls(
            classification_logits=None,
            full_classification_logits=full_logits_list,
            attention_mask=attention_mask,
            hidden_states=features,
            gate_topk_scores=gate_topk_scores,
            gate_topk_indices=gate_topk_indices,
            gate_all_scores=gate_all_scores,
            input_ids=kwargs.get("input_ids"),
            token_type_ids=token_type_ids,
        )


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------
def _head_class(version: str):
    """Resolve the head class lazily (head_seq imports this module; a module
    -level import would create a circular import)."""
    if version == "lasttok":
        return PluginGuard
    if version in ("ema", "gru"):
        from .head_seq import PluginGuardEMA, PluginGuardGRU
        return {"ema": PluginGuardEMA, "gru": PluginGuardGRU}[version]
    raise NotImplementedError(f"unsupported pguard_version: {version}")


def load_head_config(head_path: str) -> SimpleNamespace:
    with open(os.path.join(head_path, "config.json")) as f:
        raw = json.load(f)
    # PluginGuard accesses config.classifier[...] as a dict and the rest as
    # attributes, mirroring the HF config layout used during training.
    ns = SimpleNamespace(**raw)
    # runtime override for the GRU head's prefill summarization (pool | ema)
    cfc_mode = os.environ.get("PGUARD_CFC_PREFILL_MODE")
    if cfc_mode and getattr(ns, "pguard_version", "") == "gru":
        ns.prefill_mode = cfc_mode
    return ns


def build_head_from_checkpoint(head_path: str):
    """Build the safety head from a PGuard checkpoint directory (on CPU)."""
    cfg = load_head_config(head_path)
    version = cfg.pguard_version
    model = _head_class(version)(cfg)
    state = torch.load(os.path.join(head_path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("pguard_vllm: loaded head %s from %s", version, head_path)
    return cfg, model


if __name__ == "__main__":
    # in-module self-test: python -m pguard_vllm.head
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=64,
        pguard_version="lasttok",
        classifier={"classification_heads": [["task_a", 3]],
                    "dropout_prob": 0.0, "dropout_type": "stable"},
        task_schema={"task_a": {"id2label": {"0": "normal", "1": "a", "2": "b"},
                                "label2id": {"normal": 0, "a": 1, "b": 2},
                                "task_id": 0}},
        selected_layers="1,2,3", label_type="single",
    )
    head = PluginGuard(cfg)
    head.eval()
    feats = torch.randn(2, 3, 5, 64)  # (bs, n_capture_layers, len, hidden)
    with torch.no_grad():
        out = head(features=feats, token_type_ids=None, label_type=None,
                   label_marker=None)
    assert len(out.full_classification_logits) == 1
    assert tuple(out.full_classification_logits[0].shape) == (2, 5, 3)
    probs = torch.softmax(out.full_classification_logits[0], dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 5), atol=1e-5)

    # round-trip through build_head_from_checkpoint
    import tempfile
    tmp = tempfile.mkdtemp()
    torch.save(head.state_dict(), os.path.join(tmp, "pytorch_model.bin"))
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({
            "pguard_version": "lasttok", "hidden_size": 64,
            "selected_layers": "1,2,3", "label_type": "single",
            "classifier": cfg.classifier, "task_schema": cfg.task_schema,
        }, f)
    cfg2, head2 = build_head_from_checkpoint(tmp)
    assert cfg2.pguard_version == "lasttok"
    with torch.no_grad():
        out2 = head2(features=feats, token_type_ids=None, label_type=None,
                     label_marker=None)
    assert torch.allclose(out2.full_classification_logits[0],
                          out.full_classification_logits[0], atol=1e-5)

    # real checkpoints (if downloaded): strict state-dict load + forward
    pw = os.environ.get("PGUARD_PW_DIR",
                        os.path.join(os.path.dirname(tmp), "pg_weight"))
    if not os.path.isdir(pw):
        pw = "/home/zk_data/SAG-main/vllm-code/PGuard/pg_weight"
    name_map = {"lasttok": "Lasttok", "ema": "EMA", "gru": "GRU"}
    for variant in ("lasttok", "ema", "gru"):
        d = os.path.join(pw, f"PGuard-{name_map[variant]}-Qwen3.5-9B")
        if not os.path.isdir(d):
            print(f"skip {variant}: {d} not found")
            continue
        cfg3, head3 = build_head_from_checkpoint(d)
        assert cfg3.pguard_version == variant
        h = head3.to(torch.float32)
        ones = torch.ones(2, 5, dtype=torch.bool)
        feats_real = torch.randn(2, 3, 5, cfg3.hidden_size)
        with torch.no_grad():
            if variant == "gru":
                o3 = h.forward_prefill(feats_real, mask=ones)
                n = len(o3[0])
            else:
                o3 = h(features=feats_real, token_type_ids=None, label_type=None,
                       label_marker=None, attention_mask=ones)
                n = len(o3.full_classification_logits)
        assert n == len(cfg3.classifier["classification_heads"])
        print(f"{variant} checkpoint forward OK ({n} heads)")
    print("head self-test OK")
