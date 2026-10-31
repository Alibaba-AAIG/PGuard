import logging
from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.logits_processor import CaptureHiddenMode, LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models import deepseek_v4 as _dsv4
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM as SglangDeepseekV4ForCausalLM,
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
from ...patches.breakable_patch import break_graph

logger = logging.getLogger(__name__)


class DeepseekV4PGuardForCausalLM(SglangDeepseekV4ForCausalLM):

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        from sglang.srt.server_args import get_global_server_args

        base_model_path = get_global_server_args().model_path
        self.guard_model_dir = resolve_guard_model_dir(base_model_path)
        self.guard_config = read_stateless_guard_config(self.guard_model_dir)
        validate_guard_base_config(
            self.guard_config,
            config,
            str(base_model_path),
            ("deepseek_v4",),
        )
        self.capture_layer_ids = tuple(layer - 1 for layer in self.guard_config.selected_layers)
        requested = tuple(capture_layer_indices())
        if requested and requested != self.capture_layer_ids:
            raise ValueError(
                "PGUARD_CAPTURE_LAYERS conflicts with guard selected_layers: "
                f"env={requested}, expected={self.capture_layer_ids}"
            )
        self.gate_topk = self._setup_capture_layers()
        self.repeat_logits_processor = LogitsProcessor(config)
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.feature_classifier = PluginGuard(
            self.guard_config,
            hidden_size=self.guard_config.hidden_size,
            classification_heads=self.guard_config.classification_heads,
        )
        self._guard_loaded = False

    def _setup_capture_layers(self) -> bool:
        """Configure global DSpark layer IDs matching training hidden-state indices."""
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
        self.set_dspark_layers_to_capture(list(self.capture_layer_ids))
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
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.dsa_enable_prefill_cp:
            if _dsv4.can_dsa_cp_split(
                len(input_ids), self.cp_size, True, forward_batch
            ):
                forward_batch.attn_cp_metadata = (
                    _dsv4.prepare_context_parallel_metadata(
                        len(input_ids),
                        self.cp_rank,
                        self.cp_size,
                        forward_batch.seq_lens_cpu.tolist(),
                        extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                    )
                )
                if _dsv4.is_dsa_prefill_cp_round_robin_split():
                    attn_backend = _dsv4.get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        with _dsv4.get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states

        merged_features = self._merge_guard_features(aux_hidden_states, forward_batch)
        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
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
            skip_first=self.gate_topk and len(aux_hidden_states) >= 2,
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
                break_graph()

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


class DeepseekV4ForCausalLM(DeepseekV4PGuardForCausalLM):
    pass


