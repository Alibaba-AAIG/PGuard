from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertIntermediate,
    BertLayer,
    BertOutput,
)

from .guard_utils import PRODUCTION_CLASSIFICATION_HEADS, PRODUCTION_LABELS


@dataclass(frozen=True)
class SeqGuardVersionProfile:

    version: str
    gate_first: bool
    n_fusion_layers: int
    n_adapter_layers: int

    @property
    def pguard_version(self) -> str:
        return self.version

    @property
    def n_bert_layers(self) -> int:
        return self.n_fusion_layers


_SEQ_GUARD_PROFILES = {
    "ema": SeqGuardVersionProfile("ema", True, 1, 1),
}
_SEQ_SEMANTIC_REQUIRED_FIELDS = (
    "pguard_version",
    "d_feat",
    "hidden_size",
    "n_cap",
    "window_n",
    "ema_update_mask",
    "ema_scales",
    "n_bert_layers",
    "n_adapter_layers",
    "n_attn_heads",
    "top_L",
    "gate_first",
    "num_classes",
)
# GLM callers use these names; both backbones share one profile type and table.
SequenceGuardProfile = SeqGuardVersionProfile
SEQUENCE_GUARD_PROFILES = _SEQ_GUARD_PROFILES


def canonical_seq_guard_version(version: str) -> str:
    value = str(version or "").strip().lower()
    if value not in _SEQ_GUARD_PROFILES:
        supported = ", ".join(sorted(_SEQ_GUARD_PROFILES))
        raise ValueError(
            f"unknown sequence guard version {version!r}; supported: {supported}"
        )
    return value


def resolve_checkpoint_path(path: str) -> str:
    """Resolve either a checkpoint file or a directory containing pytorch_model.bin."""
    return os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else path


def load_seq_guard_config(state_path: str) -> Dict[str, Any]:
    """Read the config.json adjacent to a sequence checkpoint, if present."""
    if not state_path:
        return {}
    checkpoint_path = resolve_checkpoint_path(state_path)
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"sequence guard config must be a JSON object: {config_path}")
    return config


def resolve_seq_guard_profile(
    state_path: str = "",
    explicit_version: str = "",
) -> SeqGuardVersionProfile:
    """Resolve architecture with checkpoint metadata taking precedence.

    ``PGUARD_VERSION`` is an assertion when config.json carries ``pguard_version``
    and the selector otherwise; only ``ema`` is supported either way.
    """
    return resolve_sequence_guard_profile(
        load_seq_guard_config(state_path),
        explicit_version=explicit_version,
    )


read_head_config = load_seq_guard_config


def require_seq_guard_config(state_path: str, context: str) -> Dict[str, Any]:
    config = load_seq_guard_config(state_path)
    if state_path and not config:
        raise ValueError(
            f"{context} requires an adjacent config.json; bare state files cannot "
            "prove schema, model identity, feature selection, or EMA semantics"
        )
    return config


def _metadata_bool(value, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{source} must be a boolean, got {value!r}")


def resolve_sequence_guard_profile(
    head_config: dict,
    explicit_version: str = "",
) -> SeqGuardVersionProfile:
    """Resolve the shared profile and validate structured architecture metadata."""
    if not isinstance(head_config, dict):
        raise ValueError("sequence guard config must be an object")
    metadata_version = head_config.get("pguard_version")
    explicit = (
        canonical_seq_guard_version(explicit_version) if explicit_version else ""
    )
    metadata = (
        canonical_seq_guard_version(metadata_version) if metadata_version else ""
    )
    if metadata:
        if explicit and explicit != metadata:
            raise ValueError(
                f"PGUARD_VERSION={explicit!r} conflicts with checkpoint "
                f"config.json pguard_version={metadata!r}"
            )
        profile = _SEQ_GUARD_PROFILES[metadata]
    elif explicit:
        profile = _SEQ_GUARD_PROFILES[explicit]
    else:
        raise ValueError(
            "sequence checkpoint has no config.json pguard_version and PGUARD_VERSION "
            "is unset; add pguard_version metadata or set PGUARD_VERSION=ema"
        )

    seq_meta = head_config.get("pguard_seq") or {}
    if not isinstance(seq_meta, dict):
        raise ValueError("config.json.pguard_seq must be an object")
    nested_version = seq_meta.get("pguard_version")
    if nested_version and canonical_seq_guard_version(nested_version) != profile.version:
        raise ValueError(
            f"config.json.pguard_seq.pguard_version={nested_version!r} conflicts with "
            f"resolved profile {profile.version!r}"
        )
    for key, expected in (
        ("gate_first", profile.gate_first),
        ("n_bert_layers", profile.n_fusion_layers),
        ("n_adapter_layers", profile.n_adapter_layers),
    ):
        if key not in seq_meta:
            continue
        actual = (
            _metadata_bool(seq_meta[key], f"config.json.pguard_seq.{key}")
            if key == "gate_first"
            else int(seq_meta[key])
        )
        if actual != expected:
            raise ValueError(
                f"config.json.pguard_seq.{key}={actual!r} conflicts with "
                f"{profile.version} profile value {expected!r}"
            )
    return profile


def resolve_classification_heads(head_config: dict):
    """Validate a checkpoint's multi-task schema; None uses the caller's single head."""
    classifier = head_config.get("classifier", {})
    if not isinstance(classifier, dict):
        raise ValueError("config.json.classifier must be an object")
    heads = classifier.get("classification_heads")
    if not heads:
        return None
    if not isinstance(heads, (list, tuple)):
        raise ValueError("config.json.classifier.classification_heads must be a list")
    resolved = []
    for item in heads:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"invalid classification head entry: {item!r}")
        name, num_classes = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid classification head name: {name!r}")
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise ValueError(
                f"invalid class count for classification head {name!r}: {num_classes!r}"
            )
        resolved.append((name, num_classes))
    if len({name for name, _ in resolved}) != len(resolved):
        raise ValueError(f"duplicate classification head names: {resolved!r}")
    seq_meta = head_config.get("pguard_seq") or {}
    if not isinstance(seq_meta, dict):
        raise ValueError("config.json.pguard_seq must be an object")
    declared = seq_meta.get("num_classes")
    total = sum(n for _, n in resolved)
    if declared is not None and int(declared) != total:
        raise ValueError(
            f"classification heads total {total} classes, but "
            f"pguard_seq.num_classes={declared!r}"
        )
    return resolved


def validate_task_schema(
    head_config: Mapping[str, Any],
    classification_heads: Sequence[Tuple[str, int]],
    context: str = "sequence checkpoint",
    expected_labels: Optional[Mapping[str, Sequence[str]]] = None,
) -> None:
    """Validate task IDs and the label-ID bijection when task_schema is present."""
    task_schema = head_config.get("task_schema")
    if task_schema is None:
        return
    if not isinstance(task_schema, dict):
        raise ValueError(f"{context} task_schema must be an object")
    expected_names = {name for name, _ in classification_heads}
    if set(task_schema) != expected_names:
        raise ValueError(
            f"{context} task_schema tasks {set(task_schema)!r} do not match "
            f"classification heads {expected_names!r}"
        )
    for task_id, (task_name, num_classes) in enumerate(classification_heads):
        schema = task_schema[task_name]
        if not isinstance(schema, dict):
            raise ValueError(f"{context} task_schema.{task_name} must be an object")
        actual_task_id = schema.get("task_id")
        if isinstance(actual_task_id, bool) or actual_task_id != task_id:
            raise ValueError(
                f"{context} task_schema.{task_name}.task_id={actual_task_id!r}; "
                f"expected {task_id} from classification-head order"
            )
        label2id = schema.get("label2id")
        id2label = schema.get("id2label")
        if not isinstance(label2id, dict) or not isinstance(id2label, dict):
            raise ValueError(
                f"{context} task_schema.{task_name} must contain label2id/id2label objects"
            )
        try:
            normalized_label2id = {
                str(label): int(label_id) for label, label_id in label2id.items()
            }
            normalized_id2label = {
                int(label_id): str(label) for label_id, label in id2label.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} task_schema.{task_name} contains invalid label IDs"
            ) from exc
        expected_ids = set(range(num_classes))
        if (
            set(normalized_label2id.values()) != expected_ids
            or set(normalized_id2label) != expected_ids
            or any(
                normalized_id2label[label_id] != label
                for label, label_id in normalized_label2id.items()
            )
        ):
            raise ValueError(
                f"{context} task_schema.{task_name} label2id/id2label must be inverse "
                f"mappings over IDs 0..{num_classes - 1}"
            )
        if expected_labels is not None:
            labels = tuple(expected_labels[task_name])
            expected_label2id = {label: label_id for label_id, label in enumerate(labels)}
            if normalized_label2id != expected_label2id:
                raise ValueError(
                    f"{context} task_schema.{task_name} label-ID mapping differs from "
                    f"the production schema: got {normalized_label2id!r}, "
                    f"expected {expected_label2id!r}"
                )


def validate_production_classification_schema(
    head_config: Mapping[str, Any], context: str = "sequence checkpoint"
) -> Sequence[Tuple[str, int]]:
    heads = resolve_classification_heads(dict(head_config))
    if tuple(heads or ()) != PRODUCTION_CLASSIFICATION_HEADS:
        raise ValueError(
            f"{context} must use the exact production nine-task/38-class schema; "
            f"got {heads!r}"
        )
    if "task_schema" not in head_config:
        raise ValueError(f"{context} must include the production task_schema")
    validate_task_schema(
        head_config,
        heads,
        context=context,
        expected_labels=PRODUCTION_LABELS,
    )
    return heads


def validate_seq_model_identity(
    head_config: Mapping[str, Any],
    expected_model_types: Sequence[str],
    context: str = "sequence checkpoint",
    base_model_path: Optional[str] = None,
) -> None:
    feature_extractor = head_config.get("feature_extractor")
    if not isinstance(feature_extractor, str) or not feature_extractor.strip():
        raise ValueError(f"{context} feature_extractor is required")
    model_type = str(head_config.get("model_type", "")).strip()
    if model_type not in set(expected_model_types):
        raise ValueError(
            f"{context} model_type={model_type!r}; expected one of "
            f"{tuple(expected_model_types)!r}"
        )
    if base_model_path:
        normalize = lambda value: re.sub(
            r"[^a-z0-9]", "", os.path.basename(str(value).rstrip("/")).lower()
        )
        runtime_identity = os.getenv("PGUARD_BASE_MODEL_ID") or base_model_path
        if normalize(feature_extractor) != normalize(runtime_identity):
            raise ValueError(
                f"{context} feature_extractor={feature_extractor!r} does not match "
                f"runtime model identity {runtime_identity!r}; set PGUARD_BASE_MODEL_ID "
                "to the canonical training model ID when using a renamed local mirror"
            )


def validate_seq_feature_selection(
    head_config: Mapping[str, Any],
    selected_layers: Sequence[int],
    context: str = "sequence checkpoint",
) -> None:
    configured_layers = resolve_selected_layers(dict(head_config))
    expected_layers = tuple(int(layer) for layer in selected_layers)
    if head_config.get("pguard_capture_layers") is None:
        raise ValueError(f"{context} pguard_capture_layers is required")
    if configured_layers != expected_layers:
        raise ValueError(
            f"{context} selected_layers={configured_layers!r}; expected {expected_layers!r}"
        )


def _normalize_seq_semantic_value(key: str, value: Any, source: str) -> Any:
    if key == "pguard_version":
        return canonical_seq_guard_version(value)
    if key == "gate_first":
        return _metadata_bool(value, f"{source}.{key}")
    if key == "ema_scales":
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{source}.{key} must be a non-empty sequence")
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}.{key} contains invalid scales: {value!r}") from exc
    if key in {
        "d_feat",
        "hidden_size",
        "n_cap",
        "window_n",
        "n_bert_layers",
        "n_adapter_layers",
        "n_attn_heads",
        "top_L",
        "num_classes",
    }:
        if isinstance(value, bool):
            raise ValueError(f"{source}.{key} must be an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}.{key} must be an integer, got {value!r}") from exc
    return value


def validate_seq_guard_semantics(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    source: str,
    required_fields: Sequence[str] = (),
) -> None:
    """Compare non-tensor sequence semantics against the constructed runtime guard."""
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source} must be an object")
    missing = [key for key in required_fields if key not in metadata]
    if missing:
        raise ValueError(f"{source} is missing required semantic metadata: {missing}")
    for key, expected_value in expected.items():
        if key not in metadata or metadata[key] is None:
            continue
        actual = _normalize_seq_semantic_value(key, metadata[key], source)
        normalized_expected = _normalize_seq_semantic_value(
            key, expected_value, "runtime sequence guard"
        )
        if actual != normalized_expected:
            raise ValueError(
                f"{source}.{key}={actual!r} conflicts with runtime value "
                f"{normalized_expected!r}"
            )


def validate_seq_checkpoint_semantics(
    head_config: Mapping[str, Any],
    payload_meta: Optional[Mapping[str, Any]],
    expected: Mapping[str, Any],
    context: str = "sequence checkpoint",
    require_formal_config_metadata: bool = False,
) -> None:
    """Validate config/pguard_seq and wrapped payload metadata through one policy."""
    if not isinstance(head_config, Mapping):
        raise ValueError(f"{context} config must be an object")
    if require_formal_config_metadata and not head_config.get("pguard_version"):
        raise ValueError(
            f"{context} requires config.json pguard_version and semantic metadata"
        )
    seq_meta = head_config.get("pguard_seq") or {}
    if not isinstance(seq_meta, Mapping):
        raise ValueError(f"{context} config.json.pguard_seq must be an object")
    formal_config = bool(head_config.get("pguard_version"))
    validate_seq_guard_semantics(
        seq_meta,
        expected,
        f"{context} config.json.pguard_seq",
    )
    if "hidden_size" in head_config:
        validate_seq_guard_semantics(
            {"hidden_size": head_config["hidden_size"]},
            expected,
            f"{context} config.json",
        )
    if payload_meta:
        validate_seq_guard_semantics(
            payload_meta,
            expected,
            f"{context} payload meta",
            required_fields=(
                _SEQ_SEMANTIC_REQUIRED_FIELDS
                if require_formal_config_metadata and formal_config
                else ()
            ),
        )


def _make_bert_config(hidden_size: int, intermediate_size: int, num_attention_heads: int) -> BertConfig:
    cfg = BertConfig()
    cfg.hidden_size = hidden_size
    cfg.intermediate_size = intermediate_size
    cfg.num_attention_heads = num_attention_heads
    cfg.add_cross_attention = False
    cfg.is_decoder = False
    cfg._attn_implementation = "sdpa"
    return cfg


@dataclass(frozen=True)
class FusionTpContext:
    rank: int
    world: int
    device_group: object


@dataclass(frozen=True)
class _TpLayerWeights:
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    attention_output_weight: torch.Tensor
    intermediate_weight: torch.Tensor
    intermediate_bias: torch.Tensor
    output_weight: torch.Tensor


def _tp_all_reduce_sum(x: torch.Tensor, context: FusionTpContext) -> torch.Tensor:
    x = x.contiguous()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=context.device_group)
    return x


def _bert_layer_last_position(
    layer: BertLayer,
    x: torch.Tensor,
    qkv_w=None,
    qkv_b=None,
) -> torch.Tensor:
    sa = layer.attention.self
    B = x.size(0)
    q_shape = (B, 1, sa.num_attention_heads, sa.attention_head_size)
    kv_shape = (B, -1, sa.num_attention_heads, sa.attention_head_size)
    if qkv_w is not None:
        H = qkv_w.size(0) // 3
        q = F.linear(x[:, -1:], qkv_w[:H], qkv_b[:H]).view(*q_shape).transpose(1, 2)
        k, v = F.linear(x, qkv_w[H:], qkv_b[H:]).split(H, dim=-1)
        k = k.view(*kv_shape).transpose(1, 2)
        v = v.view(*kv_shape).transpose(1, 2)
    else:
        q = sa.query(x[:, -1:]).view(*q_shape).transpose(1, 2)
        k = sa.key(x).view(*kv_shape).transpose(1, 2)
        v = sa.value(x).view(*kv_shape).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v, scale=getattr(sa, "scaling", None))
    attn = attn.transpose(1, 2).reshape(B, 1, -1)
    h = layer.attention.output(attn, x[:, -1:])
    h = layer.output(layer.intermediate(h), h)
    return h.squeeze(1)


def _bert_layer_fused(
    layer: BertLayer,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
) -> torch.Tensor:
    sa = layer.attention.self
    B, S, _ = x.shape
    qkv = F.linear(x, qkv_w, qkv_b).view(
        B, S, 3, sa.num_attention_heads, sa.attention_head_size
    )
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
    attn = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, scale=getattr(sa, "scaling", None)
    )
    attn = attn.transpose(1, 2).reshape(B, S, -1)
    h = layer.attention.output(attn, x)
    return layer.output(layer.intermediate(h), h)


def _tp_bert_output(
    partial_input: torch.Tensor,
    residual: torch.Tensor,
    dense_bias: torch.Tensor,
    output: BertOutput,
    context: FusionTpContext,
) -> torch.Tensor:
    hidden = _tp_all_reduce_sum(partial_input, context) + dense_bias
    hidden = output.dropout(hidden)
    return output.LayerNorm(hidden + residual)


def _bert_layer_tp(
    layer: BertLayer,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    weights: _TpLayerWeights,
    context: FusionTpContext,
) -> torch.Tensor:
    sa = layer.attention.self
    batch, seq_len, _ = x.shape
    local_heads = sa.num_attention_heads // context.world
    local_hidden = local_heads * sa.attention_head_size
    qkv = F.linear(x, weights.qkv_weight, weights.qkv_bias).view(
        batch, seq_len, 3, local_heads, sa.attention_head_size
    )
    q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        scale=getattr(sa, "scaling", None),
    )
    attn = attn.transpose(1, 2).reshape(batch, seq_len, local_hidden)
    attention_partial = F.linear(attn, weights.attention_output_weight)
    hidden = _tp_bert_output(
        attention_partial,
        x,
        layer.attention.output.dense.bias,
        layer.attention.output,
        context,
    )
    intermediate = layer.intermediate.intermediate_act_fn(
        F.linear(hidden, weights.intermediate_weight, weights.intermediate_bias)
    )
    output_partial = F.linear(intermediate, weights.output_weight)
    return _tp_bert_output(
        output_partial,
        hidden,
        layer.output.dense.bias,
        layer.output,
        context,
    )


def _bert_layer_last_position_tp(
    layer: BertLayer,
    x: torch.Tensor,
    weights: _TpLayerWeights,
    context: FusionTpContext,
) -> torch.Tensor:
    sa = layer.attention.self
    batch = x.size(0)
    local_heads = sa.num_attention_heads // context.world
    local_hidden = local_heads * sa.attention_head_size
    q = F.linear(
        x[:, -1:],
        weights.qkv_weight[:local_hidden],
        weights.qkv_bias[:local_hidden],
    ).view(batch, 1, local_heads, sa.attention_head_size).transpose(1, 2)
    k, v = F.linear(
        x,
        weights.qkv_weight[local_hidden:],
        weights.qkv_bias[local_hidden:],
    ).split(local_hidden, dim=-1)
    k = k.view(batch, -1, local_heads, sa.attention_head_size).transpose(1, 2)
    v = v.view(batch, -1, local_heads, sa.attention_head_size).transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q, k, v, scale=getattr(sa, "scaling", None)
    )
    attn = attn.transpose(1, 2).reshape(batch, 1, local_hidden)
    attention_partial = F.linear(attn, weights.attention_output_weight)
    hidden = _tp_bert_output(
        attention_partial,
        x[:, -1:],
        layer.attention.output.dense.bias,
        layer.attention.output,
        context,
    )
    intermediate = layer.intermediate.intermediate_act_fn(
        F.linear(hidden, weights.intermediate_weight, weights.intermediate_bias)
    )
    output_partial = F.linear(intermediate, weights.output_weight)
    return _tp_bert_output(
        output_partial,
        hidden,
        layer.output.dense.bias,
        layer.output,
        context,
    ).squeeze(1)


class _AdapterBlock(nn.Module):

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.intermediate(x), x)


class _BasicClassificationHead(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


class SequencePluginGuard(nn.Module):

    def __init__(
        self,
        d_feat: int,
        window_n: int,
        n_cap: int,
        n_scales: int = 4,
        n_bert_layers: int = 1,
        n_adapter_layers: int = 1,
        n_attn_heads: int = 8,
        top_L: int = 3,
        num_classes: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        gate_topk: bool = True,
        fuse_kernels: bool = False,
        fuse_classifiers: bool = False,
        classification_heads: Optional[Sequence[Tuple[str, int]]] = None,
    ) -> None:
        super().__init__()
        assert d_feat % n_cap == 0, f"d_feat {d_feat} not divisible by n_cap {n_cap}"
        if n_bert_layers <= 0 or n_adapter_layers <= 0:
            raise ValueError(
                f"fusion and adapter layer counts must be positive, got "
                f"{n_bert_layers}/{n_adapter_layers}"
            )
        self.d_feat = d_feat
        self.N = window_n
        self.n_cap = n_cap
        self.hidden = d_feat // n_cap
        self.n_scales = n_scales
        self.n_fusion_layers = n_bert_layers
        self.n_adapter_layers = n_adapter_layers
        self.classification_heads = (
            list(classification_heads) if classification_heads else None
        )
        self.num_classes = (
            sum(n for _, n in self.classification_heads)
            if self.classification_heads
            else num_classes
        )
        self.top_L = min(top_L, n_cap)
        self.gate_topk = gate_topk
        self.fuse_kernels = fuse_kernels
        self._qkv_fused = None
        self._tp_fusion_weights = {}
        # Independent from QKV fusion; single-head and training paths stay unfused.
        # Derived weights are ordinary attributes, never checkpoint parameters/buffers.
        self.fuse_classifiers = fuse_classifiers
        self._classifiers_fused = None
        H = self.hidden

        fusion_cfg = _make_bert_config(H, H * 4, n_attn_heads)
        self.fusion_bert_layers = nn.ModuleList([BertLayer(fusion_cfg) for _ in range(n_bert_layers)])
        # [2] norm
        self.norm = nn.LayerNorm(H, eps=fusion_cfg.layer_norm_eps)
        self.gate = nn.Linear(H, 1)
        self.gate_temperature = 1.0
        self.gate_norm = nn.LayerNorm(H, eps=fusion_cfg.layer_norm_eps)
        ffn_cfg = BertConfig()
        self.input_proj = nn.Linear(H, ffn_cfg.hidden_size)
        self.adapter = nn.ModuleList(
            [_AdapterBlock(ffn_cfg) for _ in range(n_adapter_layers)]
        )
        self.classifiers = nn.ModuleList()
        if self.classification_heads:
            for name, n_cls in self.classification_heads:
                self.classifiers.add_module(
                    name, _BasicClassificationHead(ffn_cfg.hidden_size, n_cls)
                )
        else:
            self.classifiers.append(
                _BasicClassificationHead(ffn_cfg.hidden_size, num_classes)
            )

        self._causal_mask_cache = {}
        self.to(dtype=dtype)
        self.eval()

    def _invalidate_fused_caches(self) -> None:
        self._qkv_fused = None
        self._tp_fusion_weights.clear()
        self._classifiers_fused = None
        cache = getattr(self, "_causal_mask_cache", None)
        if cache is not None:
            cache.clear()

    def _apply(self, fn):
        result = super()._apply(fn)
        self._invalidate_fused_caches()
        return result

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._invalidate_fused_caches()
        return result

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._invalidate_fused_caches()
        return result

    @torch.no_grad()
    def forward(
        self,
        hist_seq: torch.Tensor,
        hist_mask: torch.Tensor,
        ema: torch.Tensor,
        current: torch.Tensor,
        tp_fusion_context: Optional[FusionTpContext] = None,
    ) -> torch.Tensor:
        B = current.size(0)
        w_dtype = self.norm.weight.dtype
        ema = ema.to(w_dtype).reshape(B, self.n_scales, self.n_cap, self.hidden)
        current = current.to(w_dtype).reshape(B, self.n_cap, self.hidden)

        idx, w = self._gate_weights(current)
        if idx is None:
            merged_ema = torch.einsum('bc,bsch->bsh', w, ema)     # [B, S, H]
            merged_cur = torch.einsum('bc,bch->bh', w, current)   # [B, H]
        else:
            cur_idx = idx.unsqueeze(-1).expand(-1, -1, self.hidden)      # [B,K,H]
            merged_cur = (torch.gather(current, 1, cur_idx) * w.unsqueeze(-1)).sum(dim=1)
            ema_idx = idx.unsqueeze(1).unsqueeze(-1).expand(
                -1, self.n_scales, -1, self.hidden
            )                                                            # [B,S,K,H]
            merged_ema = (
                torch.gather(ema, 2, ema_idx) * w.unsqueeze(1).unsqueeze(-1)
            ).sum(dim=2)                                                 # [B,S,H]
        merged_seq = torch.cat([merged_ema, merged_cur.unsqueeze(1)], dim=1)  # [B, S+1, H]
        merged = self.norm(
            self._fusion_forward(merged_seq, tp_fusion_context)
        )                                                                       # [B, hidden]
        # [4] FFN
        h = self.input_proj(merged)
        for layer in self.adapter:
            h = layer(h)
        return self._classify(h).float()

    def _classify(self, h: torch.Tensor) -> torch.Tensor:
        if self.fuse_classifiers and len(self.classifiers) > 1 and not self.training:
            return self._classifiers_forward_fused(h)
        return self._classify_unfused(h)

    def _classify_unfused(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.softmax(head(h), dim=-1) for head in self.classifiers], dim=-1)

    @torch.no_grad()
    def _classifiers_forward_fused(self, h: torch.Tensor) -> torch.Tensor:
        cache = self._classifiers_fused
        if cache is None or cache[0].device != h.device or cache[0].dtype != h.dtype:
            dense_w = torch.cat([head.dense.weight for head in self.classifiers], dim=0)
            dense_b = torch.cat([head.dense.bias for head in self.classifiers], dim=0)
            output_w = torch.block_diag(
                *[head.out_proj.weight for head in self.classifiers]
            )
            output_b = torch.cat(
                [head.out_proj.bias for head in self.classifiers], dim=0
            )
            self._classifiers_fused = (dense_w, dense_b, output_w, output_b)
        dense_w, dense_b, output_w, output_b = self._classifiers_fused
        first_head = next(iter(self.classifiers))
        hidden = first_head.act_fn(F.linear(h, dense_w, dense_b))
        logits = F.linear(hidden, output_w, output_b)
        pieces = logits.split([head.out_proj.out_features for head in self.classifiers], dim=-1)
        return torch.cat([torch.softmax(piece, dim=-1) for piece in pieces], dim=-1)


    def _fusion_forward(
        self,
        stacked: torch.Tensor,
        tp_fusion_context: Optional[FusionTpContext] = None,
    ) -> torch.Tensor:
        attn_mask = self._get_causal_mask(stacked)
        if tp_fusion_context is not None:
            weights = self._get_tp_fusion_weights(tp_fusion_context)
            for layer, local_weights in zip(
                self.fusion_bert_layers[:-1], weights[:-1]
            ):
                stacked = _bert_layer_tp(
                    layer, stacked, attn_mask, local_weights, tp_fusion_context
                )
            return _bert_layer_last_position_tp(
                self.fusion_bert_layers[-1],
                stacked,
                weights[-1],
                tp_fusion_context,
            )
        if self.fuse_kernels:
            if self._qkv_fused is None:
                self._build_qkv_fused()
            for i, layer in enumerate(self.fusion_bert_layers[:-1]):
                stacked = _bert_layer_fused(layer, stacked, attn_mask, *self._qkv_fused[i])
            return _bert_layer_last_position(
                self.fusion_bert_layers[-1], stacked, *self._qkv_fused[-1]
            )
        for layer in self.fusion_bert_layers[:-1]:
            out = layer(stacked, attention_mask=attn_mask)
            stacked = out[0] if isinstance(out, (tuple, list)) else out
        return _bert_layer_last_position(self.fusion_bert_layers[-1], stacked)

    def _build_qkv_fused(self) -> None:
        fused = []
        for layer in self.fusion_bert_layers:
            sa = layer.attention.self
            w = torch.cat([sa.query.weight, sa.key.weight, sa.value.weight], dim=0)
            b = torch.cat([sa.query.bias, sa.key.bias, sa.value.bias], dim=0)
            fused.append((w, b))
        self._qkv_fused = fused

    def _get_tp_fusion_weights(
        self, context: FusionTpContext
    ) -> Tuple[_TpLayerWeights, ...]:
        if context.world <= 1 or not 0 <= context.rank < context.world:
            raise ValueError(
                f"invalid fusion TP context rank={context.rank}, world={context.world}"
            )
        key = (context.rank, context.world)
        cached = self._tp_fusion_weights.get(key)
        if cached is not None:
            return cached
        local_layers = []
        for layer in self.fusion_bert_layers:
            sa = layer.attention.self
            if sa.num_attention_heads % context.world != 0:
                raise ValueError(
                    f"attention heads {sa.num_attention_heads} not divisible by "
                    f"fusion TP world {context.world}"
                )
            hidden = sa.all_head_size
            intermediate = layer.intermediate.dense.out_features
            if hidden % context.world or intermediate % context.world:
                raise ValueError(
                    f"hidden/intermediate {hidden}/{intermediate} not divisible by "
                    f"fusion TP world {context.world}"
                )
            hidden_start = context.rank * (hidden // context.world)
            hidden_end = hidden_start + hidden // context.world
            intermediate_start = context.rank * (intermediate // context.world)
            intermediate_end = intermediate_start + intermediate // context.world
            local_layers.append(
                _TpLayerWeights(
                    qkv_weight=torch.cat(
                        [
                            sa.query.weight[hidden_start:hidden_end],
                            sa.key.weight[hidden_start:hidden_end],
                            sa.value.weight[hidden_start:hidden_end],
                        ],
                        dim=0,
                    ).contiguous(),
                    qkv_bias=torch.cat(
                        [
                            sa.query.bias[hidden_start:hidden_end],
                            sa.key.bias[hidden_start:hidden_end],
                            sa.value.bias[hidden_start:hidden_end],
                        ],
                        dim=0,
                    ).contiguous(),
                    attention_output_weight=layer.attention.output.dense.weight[
                        :, hidden_start:hidden_end
                    ].contiguous(),
                    intermediate_weight=layer.intermediate.dense.weight[
                        intermediate_start:intermediate_end
                    ].contiguous(),
                    intermediate_bias=layer.intermediate.dense.bias[
                        intermediate_start:intermediate_end
                    ].contiguous(),
                    output_weight=layer.output.dense.weight[
                        :, intermediate_start:intermediate_end
                    ].contiguous(),
                )
            )
        result = tuple(local_layers)
        self._tp_fusion_weights[key] = result
        return result

    def _get_causal_mask(self, stacked: torch.Tensor) -> torch.Tensor:
        seq_len = self.n_scales + 1
        key = (stacked.device, stacked.dtype)
        attn_mask = self._causal_mask_cache.get(key)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), torch.finfo(stacked.dtype).min,
                    device=stacked.device, dtype=stacked.dtype,
                ),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            self._causal_mask_cache[key] = attn_mask
        return attn_mask

    def _gate_weights(self, features: torch.Tensor):
        scores = self.gate(self.gate_norm(features)).squeeze(-1)  # [B, n_cap]
        if self.gate_temperature != 1.0:
            scores = scores / self.gate_temperature
        if not self.gate_topk or self.n_cap <= self.top_L:
            return None, torch.softmax(scores, dim=1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_L, dim=1)
        return topk_indices, torch.softmax(topk_scores, dim=1)


def _remap_training_keys(state_dict: dict) -> dict:
    """Map training checkpoint names to inference names and reject collisions."""
    remapped = {}
    origins = {}
    for original_key, value in state_dict.items():
        key = original_key
        if key.startswith("feature_classifier."):
            key = key[len("feature_classifier."):]
        if key.startswith("ema_fusion.bert_layers."):
            key = "fusion_bert_layers." + key[len("ema_fusion.bert_layers."):]
        if key in remapped:
            raise ValueError(
                f"sequence checkpoint key remap collision: {origins[key]!r} and "
                f"{original_key!r} both map to {key!r}"
            )
        remapped[key] = value
        origins[key] = original_key
    return remapped


def _extract_seq_state_dict(payload: object, path: str) -> Tuple[dict, dict]:
    """Accept generator payloads and training-side raw state_dict checkpoints."""
    if not isinstance(payload, dict):
        raise TypeError(f"sequence checkpoint must contain a mapping: {path}")
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            raise TypeError(f"sequence checkpoint meta must be a mapping: {path}")
    else:
        state_dict = payload
        meta = {}
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"sequence checkpoint state_dict is empty or invalid: {path}")
    invalid_entries = [
        key
        for key, value in state_dict.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid_entries:
        raise TypeError(
            f"sequence checkpoint contains invalid state entries: {invalid_entries[:3]}"
        )
    return state_dict, meta


def load_seq_guard_state(guard: SequencePluginGuard, path: str, strict: bool = True) -> dict:
    """Load a wrapped or raw training checkpoint with strict key remapping."""
    checkpoint_path = resolve_checkpoint_path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state_dict, meta = _extract_seq_state_dict(payload, checkpoint_path)
    state_dict = _remap_training_keys(raw_state_dict)
    target = next(guard.parameters())
    migrated = {
        key: value.to(dtype=target.dtype, device=target.device)
        for key, value in state_dict.items()
    }
    guard.load_state_dict(migrated, strict=strict)
    return meta


def _parse_layer_metadata(value: Any, source: str) -> Tuple[int, ...]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source} must be a list or comma-separated string")
    try:
        layers = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid layers in {source}: {value!r}") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError(f"invalid layers in {source}: {layers!r}")
    if len(set(layers)) != len(layers):
        raise ValueError(f"{source} contains duplicates: {layers!r}")
    return layers


def resolve_selected_layers(head_config: dict) -> Tuple[int, ...]:
    """Parse selected_layers from a direct field or serialized TrainArguments."""
    if not isinstance(head_config, dict):
        raise ValueError("sequence guard config must be an object")
    if head_config.get("pguard_capture_layers") is None:
        raise ValueError("config.json.pguard_capture_layers is required")
    return _parse_layer_metadata(
        head_config["pguard_capture_layers"], "config.json.pguard_capture_layers"
    )


def resolve_checkpoint_capture_layers(
    head_config: Mapping[str, Any], selected_layer_offset: int = 0
) -> Tuple[int, ...]:
    """Resolve inference capture indices from shared config conventions."""
    if not isinstance(head_config, Mapping):
        raise ValueError("sequence guard config must be an object")
    seq_meta = head_config.get("pguard_seq") or {}
    if not isinstance(seq_meta, Mapping):
        raise ValueError("config.json.pguard_seq must be an object")
    if seq_meta.get("capture_layers") is not None:
        return _parse_layer_metadata(
            seq_meta["capture_layers"], "config.json.pguard_seq.capture_layers"
        )
    selected = seq_meta.get("selected_layers")
    if selected is not None:
        layers = _parse_layer_metadata(
            selected, "config.json.pguard_seq.selected_layers"
        )
    else:
        layers = resolve_selected_layers(dict(head_config))
    return tuple(layer + selected_layer_offset for layer in layers)
