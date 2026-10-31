import logging
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import CaptureHiddenMode, LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.glm4v_moe import (
    Glm4vMoeForConditionalGeneration as SglangGlm4vMoeForConditionalGeneration,
)

from ...config import (
    PGUARD_SKIP_FIRST_TOKEN,
    capture_layer_indices,
    is_debug,
)
from ...guard.guard_utils import (
    load_guard_model_strict,
    read_stateless_guard_config,
    resolve_guard_model_dir,
    validate_guard_base_config,
)
from ...guard.plugin_guard import PluginGuard
from ...patches import guard_variant

logger = logging.getLogger(__name__)


class Glm4vMoePGuardForConditionalGeneration(SglangGlm4vMoeForConditionalGeneration):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.guard_model_dir = resolve_guard_model_dir(server_args.model_path)
        self.guard_config = read_stateless_guard_config(self.guard_model_dir)
        validate_guard_base_config(
            self.guard_config,
            config,
            str(server_args.model_path),
            ("glm4v_moe", "glm4v_moe_text"),
        )
        self.capture_layer_ids = self.guard_config.selected_layers
        expected_env = tuple(layer - 1 for layer in self.capture_layer_ids)
        requested = tuple(capture_layer_indices())
        if requested and requested != expected_env:
            raise ValueError(
                "PGUARD_CAPTURE_LAYERS conflicts with guard selected_layers: "
                f"env={requested}, expected={expected_env}"
            )
        if server_args.enable_two_batch_overlap:
            raise NotImplementedError(
                "PGuard aux capture is incompatible with --enable-two-batch-overlap "
                "on GLM-4.5V"
            )
        self.gate_topk = self._setup_capture_layers()
        self.repeat_logits_processor = LogitsProcessor(text_config)
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.feature_classifier = PluginGuard(
            self.guard_config,
            hidden_size=self.guard_config.hidden_size,
            classification_heads=self.guard_config.classification_heads,
        )
        self._guard_loaded = False

    def _setup_capture_layers(self) -> bool:
        """Configure global GLM layer-entry captures on the last PP rank."""
        self.capture_aux_hidden_states = False
        if not self.pp_group.is_last_rank:
            return False
        missing = [
            layer
            for layer in self.capture_layer_ids
            if not self.model.start_layer <= layer < self.model.end_layer
        ]
        if missing:
            raise ValueError(
                "all PGuard capture layers must live on the last PP rank; "
                f"missing={missing}, last-rank range="
                f"[{self.model.start_layer}, {self.model.end_layer})"
            )
        self.model.layers_to_capture = list(self.capture_layer_ids)
        self.capture_aux_hidden_states = True
        return False

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        if not self._guard_loaded:
            load_guard_model_strict(self.feature_classifier, self.guard_model_dir)
            self._guard_loaded = True
        super().load_weights(weights=weights, is_nextn=is_nextn)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.size(0) == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.size()}"
                )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states

        if not self.pp_group.is_last_rank:
            return hidden_states
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        merged_features = self._merge_guard_features(aux_hidden_states, forward_batch)
        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
        self._attach_guard_logits(
            result, merged_features, input_ids, forward_batch, aux_hidden_states
        )
        return result

    def _merge_guard_features(
        self, aux_hidden_states, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        need_guard_logits = (
            forward_batch.forward_mode.is_decode()
            or not self.skip_first_token_guard_logits
        )
        if not need_guard_logits or not aux_hidden_states:
            return None
        features = torch.stack(list(aux_hidden_states), dim=1)
        if is_debug():
            logger.warning(f"features.shape: {features.shape}")
        return self.feature_classifier.merge_multi_layer_feature(
            features,
            skip_first=False,
            skip_last=False,
            gate_topk=self.gate_topk,
        )

    def _attach_guard_logits(
        self,
        result,
        merged_features: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        aux_hidden_states,
    ) -> None:
        last_token_hidden_states = None
        if merged_features is not None:
            if forward_batch.forward_mode.is_cuda_graph():
                last_token_hidden_states = merged_features
            else:
                old_capture_hidden_mode = forward_batch.capture_hidden_mode
                forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                tmp_result = self.repeat_logits_processor(
                    input_ids,
                    merged_features.squeeze(1),
                    self.lm_head,
                    forward_batch,
                    None,
                )
                forward_batch.capture_hidden_mode = old_capture_hidden_mode
                last_token_hidden_states = tmp_result.hidden_states.unsqueeze(1)

        guard_logits = None
        if last_token_hidden_states is not None:
            if guard_variant.enabled():
                guard_variant.break_before_guard()
            guard_logits = self.feature_classifier(features=last_token_hidden_states)
            if guard_variant.enabled():
                guard_variant.record_eager_guard_execution()

        if guard_logits is None:
            if is_debug():
                logger.error(
                    f"guard_logits is None, "
                    f"is_aux_hidden_states_none={aux_hidden_states is None}, "
                    f"{forward_batch.batch_size=}"
                )
            result.customized_info = {
                "guard_logits": [0.0] * forward_batch.batch_size
            }
            return
        assert guard_logits.shape[0] == forward_batch.batch_size, (
            f"guard_logits.shape: {guard_logits.shape} != {forward_batch.batch_size}"
        )
        if is_debug():
            logger.warning(f"{guard_logits.shape=}")
        result.customized_info = {"guard_logits": guard_logits}


class Glm4vMoeForConditionalGeneration(Glm4vMoePGuardForConditionalGeneration):
    pass


