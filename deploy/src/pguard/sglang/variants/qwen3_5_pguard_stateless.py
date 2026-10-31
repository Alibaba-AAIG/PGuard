from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Sequence, Tuple

import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration as SglangQwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration as SglangQwen3_5MoeForConditionalGeneration,
)

from ...config import PGUARD_SKIP_FIRST_TOKEN
from ...guard.guard_utils import (
    GuardModelConfig,
    load_guard_model_strict,
    read_guard_config,
    resolve_guard_model_dir,
    validate_guard_base_config,
)
from ...guard.plugin_guard import PluginGuard
from ...patches import guard_variant

logger = logging.getLogger(__name__)

def hf_hidden_state_indices_to_capture_layer_ids(
    selected_layers: Sequence[int], num_hidden_layers: int
) -> tuple[int, ...]:
    """Map training/HF representation indices to Qwen3.5 capture layer ids.

    ``prepare_attn_and_capture_last_layer_outputs`` appends ``residual`` after
    ``prepare_attn`` at the beginning of layer ``i``. That tensor is HF
    ``hidden_states[i]``. The final norm at index ``num_hidden_layers`` has no
    decoder-layer capture point and is intentionally rejected.
    """
    result = tuple(int(index) for index in selected_layers)
    invalid = [index for index in result if index < 0 or index >= num_hidden_layers]
    if invalid:
        raise ValueError(
            f"selected_layers {invalid} cannot be captured by a {num_hidden_layers}-layer "
            f"Qwen3.5 model; valid HF hidden-state indices are 0..{num_hidden_layers - 1}"
        )
    if len(set(result)) != len(result):
        raise ValueError(f"selected_layers contains duplicates: {result}")
    return result


def validate_runtime_config(
    guard: GuardModelConfig,
    base_config,
    base_model_path: str,
    capture_env: Optional[str] = None,
) -> tuple[int, ...]:
    text_config = getattr(base_config, "text_config", base_config)
    hidden_size = int(text_config.hidden_size)
    num_layers = int(text_config.num_hidden_layers)
    validate_guard_base_config(
        guard,
        base_config,
        base_model_path,
        ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"),
    )
    capture_ids = hf_hidden_state_indices_to_capture_layer_ids(
        guard.selected_layers, num_layers
    )
    if capture_env is not None and capture_env.strip():
        try:
            capture_ids = tuple(
                int(value.strip()) for value in capture_env.split(",") if value.strip()
            )
        except ValueError as exc:
            raise ValueError(f"invalid PGUARD_CAPTURE_LAYERS={capture_env!r}") from exc
    logger.info(
        "PGuard config validated: hidden=%d layers=%s heads=%d classes=%d",
        hidden_size,
        capture_ids,
        len(guard.classification_heads),
        guard.num_classes,
    )
    return capture_ids


def select_last_token_features(
    merged_features: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    """Select one feature per request without rerunning ``LogitsProcessor``."""
    if forward_batch.forward_mode.is_cuda_graph() or forward_batch.forward_mode.is_decode():
        return merged_features
    seq_lens = forward_batch.extend_seq_lens
    if not isinstance(seq_lens, torch.Tensor):
        seq_lens = torch.as_tensor(seq_lens, device=merged_features.device)
    else:
        seq_lens = seq_lens.to(device=merged_features.device)
    if seq_lens.numel() != forward_batch.batch_size:
        raise RuntimeError(
            f"extend_seq_lens has {seq_lens.numel()} entries for batch_size="
            f"{forward_batch.batch_size}"
        )
    indices = torch.cumsum(seq_lens.to(torch.long), dim=0) - 1
    if torch.any(indices < 0):
        raise RuntimeError("stateless PGuard cannot select a last token from an empty sequence")
    return merged_features.index_select(0, indices)


class _Qwen3_5StatelessMixin:
    def _init_pguard(self, outer_config, base_model_path: Optional[str] = None) -> None:
        if base_model_path is None:
            from sglang.srt.server_args import get_global_server_args

            base_model_path = get_global_server_args().model_path
        self.guard_model_dir = resolve_guard_model_dir(base_model_path)
        self.guard_config = read_guard_config(self.guard_model_dir)
        self.capture_layer_ids = validate_runtime_config(
            self.guard_config,
            outer_config,
            str(base_model_path),
            os.getenv("PGUARD_CAPTURE_LAYERS"),
        )
        self.capture_aux_hidden_states = False
        if self.pp_group.is_last_rank:
            local = set(range(self.model.start_layer, self.model.end_layer))
            missing = [layer for layer in self.capture_layer_ids if layer not in local]
            if missing:
                raise ValueError(
                    "all PGuard capture layers must live on the last PP rank; "
                    f"missing={missing}, last-rank range="
                    f"[{self.model.start_layer}, {self.model.end_layer})"
                )
            self.model.set_dflash_layers_to_capture(list(self.capture_layer_ids))
            self.capture_aux_hidden_states = True
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.feature_classifier = PluginGuard(
            self.guard_config,
            hidden_size=self.guard_config.hidden_size,
            classification_heads=self.guard_config.classification_heads,
        )
        self._guard_loaded = False

    def _load_pguard_once(self) -> None:
        if not self._guard_loaded:
            load_guard_model_strict(self.feature_classifier, self.guard_model_dir)
            self._guard_loaded = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self._load_pguard_once()
        return super().load_weights(weights=weights)

    def _merge_guard_features(
        self, aux_hidden_states, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        need_guard = (
            forward_batch.forward_mode.is_decode()
            or not self.skip_first_token_guard_logits
        )
        if not need_guard or not aux_hidden_states:
            return None
        if len(aux_hidden_states) != len(self.capture_layer_ids):
            raise RuntimeError(
                f"captured {len(aux_hidden_states)} layers, expected "
                f"{len(self.capture_layer_ids)} ({self.capture_layer_ids})"
            )
        features = torch.stack(list(aux_hidden_states), dim=1)
        return self.feature_classifier.merge_multi_layer_feature(
            features, skip_first=False, skip_last=False, gate_topk=False
        )

    def _attach_guard_logits(
        self,
        result,
        merged_features: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        guard_logits = None
        if merged_features is not None:
            last_features = select_last_token_features(merged_features, forward_batch)
            if guard_variant.enabled():
                guard_variant.break_before_guard()
            guard_logits = self.feature_classifier(features=last_features)
            if guard_variant.enabled():
                guard_variant.record_eager_guard_execution()

        if guard_logits is None:
            result.customized_info = {
                "guard_logits": [0.0] * forward_batch.batch_size
            }
            return
        if guard_logits.shape != (forward_batch.batch_size, self.guard_config.num_classes):
            raise RuntimeError(
                f"guard logits shape {tuple(guard_logits.shape)} != "
                f"({forward_batch.batch_size}, {self.guard_config.num_classes})"
            )
        result.customized_info = {"guard_logits": guard_logits}

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        # Keep the native Qwen3.5 multimodal/deepstack path byte-for-byte in spirit.
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions
        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires (3, seq_len) "
                    f"positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            if not isinstance(hidden_states, tuple) or len(hidden_states) != 2:
                raise RuntimeError("Qwen3.5 capture was enabled but returned no aux states")
            hidden_states, aux_hidden_states = hidden_states

        if not self.pp_group.is_last_rank:
            return hidden_states
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        merged = self._merge_guard_features(aux_hidden_states, forward_batch)
        # Preserve the native base-logit call exactly; guard work only consumes its result.
        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )
        self._attach_guard_logits(result, merged, forward_batch)
        return result


class Qwen3_5PGuardForConditionalGeneration(
    _Qwen3_5StatelessMixin, SglangQwen3_5ForConditionalGeneration
):
    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        self._init_pguard(config)


class Qwen3_5PGuardMoeForConditionalGeneration(
    _Qwen3_5StatelessMixin, SglangQwen3_5MoeForConditionalGeneration
):
    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        self._init_pguard(config)


class Qwen3_5ForConditionalGeneration(Qwen3_5PGuardForConditionalGeneration):
    """Architecture-name replacement for the dense Qwen3.5 model."""


class Qwen3_5MoeForConditionalGeneration(Qwen3_5PGuardMoeForConditionalGeneration):
    """Architecture-name replacement for the Qwen3.5 MoE models."""


# EntryClass is intentionally defined only by qwen3_5_pguard_sglang.py.
