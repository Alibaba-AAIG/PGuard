"""Inference form of the training-side ``lasttok`` stateless PGuard head."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertConfig, BertIntermediate, BertOutput

from ..config import (
    PGUARD_FUSE_ADAPTER,
    PGUARD_FUSE_CLASSIFIERS,
    PGUARD_FUSE_MERGE,
    PGUARD_GATE_TEMPERATURE,
    is_debug,
)

logger = logging.getLogger(__name__)


class BasicClassificationHead(nn.Module):
    """Parameter-compatible inference form of the training classification head."""

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_classes)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.act_fn(self.dense(feature)))


class AdapterBlock(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.intermediate(hidden_states), hidden_states)


class PluginGuard(nn.Module):
    """The lasttok input-norm, top-3 gating, and three-FFN head."""

    def __init__(
        self,
        config,
        hidden_size: Optional[int] = None,
        classification_heads: Optional[Iterable[tuple[str, int]]] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size if hidden_size is not None else config.hidden_size)
        if classification_heads is None:
            classifier = getattr(config, "classifier", None)
            if not isinstance(classifier, dict) or not classifier.get("classification_heads"):
                raise ValueError("classification_heads must come from the guard config")
            classification_heads = classifier["classification_heads"]
        heads = tuple((str(name), int(size)) for name, size in classification_heads)
        if not heads:
            raise ValueError("classification_heads cannot be empty")

        self.top_L = 3
        self.agg = "soft"
        self.gate_temperature = PGUARD_GATE_TEMPERATURE
        self.gate = nn.Linear(hidden_size, 1)

        bert_config = BertConfig()
        self.norm = nn.LayerNorm(hidden_size, eps=bert_config.layer_norm_eps)
        self.input_proj = nn.Linear(hidden_size, bert_config.hidden_size)
        self.adapter = nn.ModuleList([AdapterBlock(bert_config) for _ in range(3)])

        self.classifiers = nn.ModuleList()
        for name, num_classes in heads:
            self.classifiers.add_module(
                name,
                BasicClassificationHead(bert_config.hidden_size, num_classes),
            )
        self.classification_heads = heads
        self._classifier_fused = None

    def _invalidate_derived_caches(self) -> None:
        self._classifier_fused = None

    def _apply(self, fn):
        try:
            return super()._apply(fn)
        finally:
            self._invalidate_derived_caches()

    def load_state_dict(self, *args, **kwargs):
        try:
            return super().load_state_dict(*args, **kwargs)
        finally:
            self._invalidate_derived_caches()

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._invalidate_derived_caches()
        return result

    def merge_multi_layer_feature(
        self,
        features: torch.Tensor,
        *,
        skip_first: bool = False,
        skip_last: bool = False,
        gate_topk: bool = True,
    ) -> torch.Tensor:
        """Apply lasttok input normalization and soft layer gating."""
        if features.ndim != 3:
            raise ValueError(f"expected [tokens,layers,hidden], got {tuple(features.shape)}")
        if skip_first:
            features = features[:, 1:, :]
        if skip_last:
            features = features[:, :-1, :]
        if features.shape[1] == 0:
            raise ValueError("no feature layers remain after slicing")
        if features.shape[-1] != self.gate.in_features:
            raise ValueError(
                f"feature hidden size {features.shape[-1]} != guard hidden size "
                f"{self.gate.in_features}"
            )
        if features.dtype != self.gate.weight.dtype:
            features = features.to(self.gate.weight.dtype)
        features = self.norm(features)
        if features.shape[1] == 1:
            return features

        features4 = features.unsqueeze(2)
        if gate_topk:
            merged, _, _ = self.forward_gate_topk(features4)
        else:
            merged, _, _ = self.forward_gate_all(features4)
        return merged

    def _gate_scores(self, features: torch.Tensor) -> torch.Tensor:
        scores = self.gate(features).squeeze(-1)
        if self.gate_temperature != 1.0 or not PGUARD_FUSE_MERGE:
            scores = scores / self.gate_temperature
        return scores

    def forward_gate_topk(self, features: torch.Tensor):
        if features.ndim != 4:
            raise ValueError(f"expected [B,L,S,H], got {tuple(features.shape)}")
        batch, n_layers, seq_len, hidden = features.shape
        if n_layers < self.top_L:
            raise ValueError(f"gate top-k needs at least {self.top_L} layers, got {n_layers}")
        scores = self._gate_scores(features)
        top_scores, top_indices = torch.topk(scores, k=self.top_L, dim=1)
        indices = top_indices.unsqueeze(-1).expand(batch, self.top_L, seq_len, hidden)
        selected = torch.gather(features, dim=1, index=indices)
        if self.agg == "hard":
            merged = selected.mean(dim=1)
        elif self.agg == "soft":
            merged = (selected * F.softmax(top_scores, dim=1).unsqueeze(-1)).sum(dim=1)
        else:
            raise NotImplementedError(self.agg)
        return merged, top_scores, top_indices

    def forward_gate_all(self, features: torch.Tensor):
        if features.ndim != 4:
            raise ValueError(f"expected [B,L,S,H], got {tuple(features.shape)}")
        scores = self._gate_scores(features)
        if self.agg == "hard":
            merged = features.mean(dim=1)
        elif self.agg == "soft":
            weights = F.softmax(scores, dim=1)
            if PGUARD_FUSE_MERGE and features.shape[2] == 1:
                merged = torch.bmm(weights.squeeze(2).unsqueeze(1), features.squeeze(2))
            else:
                merged = (features * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise NotImplementedError(self.agg)
        return merged, scores, None

    @staticmethod
    def _forward_adapter_fused(
        layer: AdapterBlock, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        intermediate = torch.addmm(
            layer.intermediate.dense.bias,
            flat,
            layer.intermediate.dense.weight.t(),
        )
        intermediate = F.gelu(intermediate, approximate="none")
        projected = torch.addmm(
            layer.output.dense.bias,
            intermediate,
            layer.output.dense.weight.t(),
        )
        output = F.layer_norm(
            projected + flat,
            layer.output.LayerNorm.normalized_shape,
            layer.output.LayerNorm.weight,
            layer.output.LayerNorm.bias,
            layer.output.LayerNorm.eps,
        )
        return output.view(shape)

    def _run_adapter(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.adapter:
            if PGUARD_FUSE_ADAPTER and not self.training and features.is_cuda:
                features = self._forward_adapter_fused(layer, features)
            else:
                features = layer(features)
        return features

    def _classify_unfused(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.softmax(head(features), dim=-1) for head in self.classifiers], dim=-1)

    def _build_classifier_fused(self) -> None:
        heads = list(self.classifiers)
        first_weight = torch.cat([head.dense.weight for head in heads], dim=0)
        first_bias = torch.cat([head.dense.bias for head in heads], dim=0)
        max_classes = max(head.out_proj.out_features for head in heads)
        out_weight = first_weight.new_zeros(
            len(heads), max_classes, first_weight.shape[1]
        )
        out_bias = first_bias.new_zeros(len(heads), max_classes)
        for index, head in enumerate(heads):
            size = head.out_proj.out_features
            out_weight[index, :size].copy_(head.out_proj.weight)
            out_bias[index, :size].copy_(head.out_proj.bias)
        self._classifier_fused = (first_weight, first_bias, out_weight, out_bias)

    def _classify_fused(self, features: torch.Tensor) -> torch.Tensor:
        if self._classifier_fused is None:
            self._build_classifier_fused()
        first_weight, first_bias, out_weight, out_bias = self._classifier_fused
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        hidden = F.gelu(F.linear(flat, first_weight, first_bias), approximate="none")
        hidden = hidden.view(
            flat.shape[0], len(self.classifiers), shape[-1]
        ).permute(1, 0, 2)
        logits = torch.bmm(hidden, out_weight.transpose(1, 2)) + out_bias[:, None, :]
        pieces = [
            torch.softmax(logits[index, :, :num_classes], dim=-1)
            for index, (_, num_classes) in enumerate(self.classification_heads)
        ]
        return torch.cat(pieces, dim=-1).view(*shape[:-1], -1)

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        if PGUARD_FUSE_CLASSIFIERS and not self.training and len(self.classifiers) > 1:
            return self._classify_fused(features)
        return self._classify_unfused(features)

    def forward(self, features: torch.Tensor, **kwargs) -> Optional[torch.Tensor]:
        if features.ndim == 4:
            if features.shape[2] != 1:
                raise ValueError(f"stateless guard expects S=1, got {tuple(features.shape)}")
            features = features.squeeze(2)
        if features.ndim != 3 or features.shape[1] != 1:
            raise ValueError(f"expected [batch,1,hidden], got {tuple(features.shape)}")
        if is_debug():
            logger.warning(
                "guard input shape=%s dtype=%s", tuple(features.shape), features.dtype
            )

        features = self.input_proj(features)
        features = self._run_adapter(features)
        logits = self._classify(features)
        return logits.reshape(logits.shape[0], -1)
