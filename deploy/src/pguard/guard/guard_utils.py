"""Guard checkpoint configuration and strict-loading helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..config import PGUARD_MODEL_PATH

_EXPECTED_VERSION = "lasttok"
PRODUCTION_CLASSIFICATION_HEADS = (
    ("Crimes & Illegal Activities", 6),
    ("Hate Speech", 5),
    ("Physical & Mental Health", 3),
    ("Ethics & Morality", 3),
    ("Data Privacy", 3),
    ("Cybersecurity", 6),
    ("Extremism", 4),
    ("Inappropriate Suggestions", 4),
    ("Risks Involving Minors", 4),
)
PRODUCTION_LABELS = {
    "Crimes & Illegal Activities": ("normal", "Pornographic Contraband", "Drug Crimes", "Dangerous Weapons", "Property Infringement", "Economic Crimes"),
    "Hate Speech": ("normal", "Abusive Curses", "Defamation", "Threats & Intimidation", "Cyberbullying"),
    "Physical & Mental Health": ("normal", "Personal Safety", "Mental Health"),
    "Ethics & Morality": ("normal", "Social Ethics", "Science & Tech Ethics"),
    "Data Privacy": ("normal", "Personal Privacy", "Commercial Secret"),
    "Cybersecurity": ("normal", "Access Control", "Malicious Code", "Hacker Attack", "Physical Security", "Social Engineering, Phishing"),
    "Extremism": ("normal", "Violent Terrorist Activities", "Social Disruption", "Extremist Ideological Trends"),
    "Inappropriate Suggestions": ("normal", "Finance", "Medicine", "Law"),
    "Risks Involving Minors": ("normal", "Corruption of Minors", "Minor Abuse & Exploitation", "Minor Delinquency"),
}


@dataclass(frozen=True)
class GuardModelConfig:
    model_dir: Path
    hidden_size: int
    classification_heads: tuple[tuple[str, int], ...]
    pguard_version: str
    feature_extractor: str
    selected_layers: tuple[int, ...]
    raw: Mapping[str, Any]

    @property
    def classifier(self) -> dict[str, Any]:
        """Expose the legacy config attribute consumed by ``PluginGuard(config)``."""
        classifier = self.raw["classifier"]
        assert isinstance(classifier, dict)
        return classifier

    @property
    def num_classes(self) -> int:
        return sum(size for _, size in self.classification_heads)


def resolve_guard_model_dir(model_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the required guard directory without silently using random weights."""
    if PGUARD_MODEL_PATH:
        path = Path(PGUARD_MODEL_PATH).expanduser()
    elif model_path is not None:
        path = Path(model_path).expanduser() / "guard_model"
    else:
        try:
            from sglang.srt.server_args import get_global_server_args

            path = Path(get_global_server_args().model_path).expanduser() / "guard_model"
        except Exception as exc:
            raise RuntimeError(
                "PGUARD_MODEL_PATH is required when the base model path is unavailable"
            ) from exc
    if not path.is_dir():
        raise FileNotFoundError(f"PGuard model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"PGuard config.json does not exist: {path / 'config.json'}")
    return path


def _parse_selected_layers(raw: Mapping[str, Any]) -> tuple[int, ...]:
    value = raw.get("pguard_capture_layers")
    if value is None:
        raise ValueError("guard config pguard_capture_layers is required")
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        try:
            layers = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"invalid selected_layers={value!r}") from exc
    elif isinstance(value, Sequence):
        layers = tuple(int(item) for item in value)
    else:
        raise ValueError(f"invalid selected_layers type: {type(value).__name__}")
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError(f"selected_layers must be unique non-negative integers, got {layers}")
    return layers


def _validate_task_schema(
    raw: Mapping[str, Any], heads: tuple[tuple[str, int], ...], expected_labels=PRODUCTION_LABELS
) -> None:
    schema = raw.get("task_schema")
    if not isinstance(schema, Mapping) or set(schema) != {name for name, _ in heads}:
        raise ValueError("task_schema tasks must exactly match classification_heads")
    for task_id, (name, size) in enumerate(heads):
        task = schema[name]
        if not isinstance(task, Mapping) or int(task.get("task_id", -1)) != task_id:
            raise ValueError(f"task_schema[{name!r}].task_id must be {task_id}")
        id2label = task.get("id2label")
        label2id = task.get("label2id")
        if not isinstance(id2label, Mapping) or not isinstance(label2id, Mapping):
            raise ValueError(f"task_schema[{name!r}] must contain id2label and label2id")
        normalized = {int(index): str(label) for index, label in id2label.items()}
        if set(normalized) != set(range(size)):
            raise ValueError(
                f"task_schema[{name!r}].id2label must contain IDs 0..{size - 1}"
            )
        normalized_label2id = {
            str(label): int(index) for label, index in label2id.items()
        }
        if normalized_label2id != {
            label: index for index, label in normalized.items()
        }:
            raise ValueError(f"task_schema[{name!r}] label mappings are inconsistent")
        expected_label2id = {
            label: index for index, label in enumerate(expected_labels[name])
        }
        if normalized_label2id != expected_label2id:
            raise ValueError(
                f"task_schema[{name!r}] differs from the production label-ID mapping"
            )


def read_stateless_guard_config(model_dir: str | os.PathLike[str]) -> GuardModelConfig:
    """Read a production v13.2 stateless head and validate every semantic field."""
    model_dir = Path(model_dir).expanduser()
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        raw = json.load(handle)

    classifier = raw.get("classifier")
    heads_raw = classifier.get("classification_heads") if isinstance(classifier, dict) else None
    if not isinstance(heads_raw, list) or not heads_raw:
        raise ValueError("guard config classifier.classification_heads is missing or empty")
    heads: list[tuple[str, int]] = []
    for item in heads_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"invalid classification head entry: {item!r}")
        name, size = str(item[0]), int(item[1])
        if not name or size <= 0 or any(name == old for old, _ in heads):
            raise ValueError(f"invalid/duplicate classification head: {item!r}")
        heads.append((name, size))

    cfg = GuardModelConfig(
        model_dir=model_dir,
        hidden_size=int(raw["hidden_size"]),
        classification_heads=tuple(heads),
        pguard_version=str(raw.get("pguard_version", "")),
        feature_extractor=str(raw.get("feature_extractor", "")),
        selected_layers=_parse_selected_layers(raw),
        raw=raw,
    )
    if cfg.pguard_version != _EXPECTED_VERSION:
        raise ValueError(
            f"guard pguard_version mismatch: expected {_EXPECTED_VERSION!r}, "
            f"got {cfg.pguard_version!r}"
        )
    if not cfg.feature_extractor:
        raise ValueError("guard feature_extractor is required")
    if raw.get("label_type") != "single":
        raise ValueError(
            f"lasttok requires label_type='single', got {raw.get('label_type')!r}"
        )
    if raw.get("pguard_capture_layers") is None:
        raise ValueError("guard config pguard_capture_layers is required")
    if len(cfg.selected_layers) != 3:
        raise ValueError(
            f"lasttok requires exactly three selected layers, got {cfg.selected_layers}"
        )
    if cfg.classification_heads != PRODUCTION_CLASSIFICATION_HEADS:
        raise ValueError(
            "lasttok requires the exact production classification schema; "
            f"expected {PRODUCTION_CLASSIFICATION_HEADS!r}, got {cfg.classification_heads!r}"
        )
    _validate_task_schema(raw, cfg.classification_heads)
    return cfg


def read_guard_config(model_dir: str | os.PathLike[str]) -> GuardModelConfig:
    """Compatibility name for the strict production stateless config reader."""
    return read_stateless_guard_config(model_dir)


def validate_guard_base_config(
    guard: GuardModelConfig,
    base_config: Any,
    base_model_path: str,
    expected_model_types: Sequence[str],
) -> None:
    base = getattr(base_config, "text_config", base_config)
    hidden_size = int(base.hidden_size)
    if hidden_size != guard.hidden_size:
        raise ValueError(
            f"guard/base hidden_size mismatch: guard={guard.hidden_size}, base={hidden_size}"
        )
    model_type = str(guard.raw.get("model_type", ""))
    if model_type not in set(expected_model_types):
        raise ValueError(
            f"guard model_type={model_type!r}; expected one of {tuple(expected_model_types)!r}"
        )
    normalize = lambda value: re.sub(
        r"[^a-z0-9]", "", Path(str(value).rstrip("/")).name.lower()
    )
    runtime_identity = os.getenv("PGUARD_BASE_MODEL_ID") or base_model_path
    if normalize(guard.feature_extractor) != normalize(runtime_identity):
        raise ValueError(
            "guard/base feature_extractor mismatch: "
            f"guard={guard.feature_extractor!r}, runtime={runtime_identity!r}; "
            "set PGUARD_BASE_MODEL_ID to the canonical training model ID when using "
            "a renamed local mirror"
        )


def get_config_from_guard_model() -> GuardModelConfig:
    return read_stateless_guard_config(resolve_guard_model_dir())


def _unwrap_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    while isinstance(obj, Mapping):
        tensor_items = {key: value for key, value in obj.items() if isinstance(value, torch.Tensor)}
        if tensor_items and len(tensor_items) == len(obj):
            return tensor_items
        wrapped = None
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                wrapped = value
                break
        if wrapped is None:
            break
        obj = wrapped
    raise TypeError("guard checkpoint is not a tensor state_dict or a supported wrapper")


def load_guard_state_dict(model_dir: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    """Load and normalize a guard checkpoint; validation is done by strict load_state_dict."""
    model_dir = Path(model_dir).expanduser()
    safe_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safe_path.is_file():
        from safetensors.torch import load_file

        raw = load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        try:
            raw = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            raw = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"no guard weights found in {model_dir}; expected model.safetensors "
            "or pytorch_model.bin"
        )

    state = _unwrap_state_dict(raw)
    normalized: dict[str, torch.Tensor] = {}
    for original_name, tensor in state.items():
        name = str(original_name)
        while name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("feature_classifier."):
            name = name[len("feature_classifier.") :]
        if name in normalized:
            raise ValueError(
                f"duplicate guard key after prefix normalization: {original_name!r} -> {name!r}"
            )
        normalized[name] = tensor
    return normalized


def load_guard_model_strict(module: torch.nn.Module, model_dir: str | os.PathLike[str]) -> None:
    """Strictly load every guard tensor; missing, unexpected and shape errors are fatal."""
    state = load_guard_state_dict(model_dir)
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"strict PGuard weight loading failed for {model_dir}: {exc}") from exc
