import logging
import os
from typing import Iterable, Optional, Tuple
import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.glm4v_moe import (
    Glm4vMoeForConditionalGeneration as SglangGlm4vMoeForConditionalGeneration,
)
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from ...config import (
    PGUARD_PREFILL_RISK_MODE,
    PGUARD_PREFILL_SCORE_PROB,
    PGUARD_CFC_PROJ_DIM,
    PGUARD_CFC_MEM_DIM,
    PGUARD_CFC_USE_DT,
    PGUARD_CFC_DT,
    PGUARD_CFC_NUM_CLASSES,
    PGUARD_CFC_STATE_PATH,
    PGUARD_CFC_GRAPH_UNROLL,
    PGUARD_CFC_USE_TRITON,
    PGUARD_CFC_PREFILL_MODE,
    PGUARD_CFC_PREFILL_EFFECTIVE_LENGTHS,
    PGUARD_SKIP_FIRST_TOKEN,
    capture_layer_indices,
)
from ...patches import guard_variant
from ...guard.recurrent_state import RecurrentStateBuffer
from ...guard.cfc_guard import (
    CfcPluginGuard,
    load_cfc_guard_state,
    resolve_cfc_classification_heads,
    validate_cfc_guard_config,
)

logger = logging.getLogger(__name__)


def _cfc_guard_ensure_rec_buf(self, forward_batch):
    if self.rec_buf is not None:
        return
    if not self.pp_group.is_last_rank:
        return
    pool = getattr(forward_batch, "req_to_token_pool", None)
    if pool is None:
        try:
            from sglang.srt.model_executor.forward_context import get_req_to_token_pool

            pool = get_req_to_token_pool()
        except Exception as e:
            logger.warning(f"[CFC_GUARD] get_req_to_token_pool 失败: {e}")
    req_to_token = getattr(pool, "req_to_token", None)
    if req_to_token is None or req_to_token.shape[0] <= 0:
        raise RuntimeError(
            "[CFC_GUARD] 拿不到 req_to_token_pool，拒绝用猜测值分配 rec_buf：slot id 越界会触发 device-side assert。检查当前 sglang 是否提供 ForwardBatch.req_to_token_pool 或 forward_context.get_req_to_token_pool()。"
        )
    max_req = int(req_to_token.shape[0])
    device = self.lm_head.weight.device
    self.rec_buf = RecurrentStateBuffer(
        max_req=max_req,
        mem_dim=self.cfc_guard.mem_dim,
        dtype=self.lm_head.weight.dtype,
        num_classes=self.cfc_guard.num_classes,
    ).to(device)
    logger.info(
        f"[CFC_GUARD] rec_buf init: max_req={max_req}, mem_dim={self.cfc_guard.mem_dim}, device={device}"
    )


def _cfc_guard_run(self, guard_hidden_states, forward_batch):
    if not guard_hidden_states:
        return None
    is_decode = forward_batch.forward_mode.is_decode()
    is_extend = forward_batch.forward_mode.is_extend()
    if not (is_decode or is_extend):
        return None
    _cfc_guard_ensure_rec_buf(self, forward_batch)
    if self.rec_buf is None:
        return None
    feat_t = torch.stack(guard_hidden_states, dim=1)
    slot_ids = forward_batch.req_pool_indices.to(torch.long)
    if is_extend:
        seq_lens = forward_batch.extend_seq_lens.to(torch.long)
        seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if seq_lens_cpu is None:
            raise RuntimeError(
                "[CFC_GUARD] extend forward 缺少 extend_seq_lens_cpu，无法在不触发 GPU host sync 的前提下执行 prefill"
            )
        seq_lens_host = [int(length) for length in seq_lens_cpu]
        if (
            len(seq_lens_host) != forward_batch.batch_size
            or any((length <= 0 for length in seq_lens_host))
            or sum(seq_lens_host) != feat_t.size(0)
        ):
            raise ValueError(
                f"[CFC_GUARD] extend_seq_lens_cpu 与 packed features 不一致: batch={forward_batch.batch_size}, lengths={seq_lens_host}, tokens={feat_t.size(0)}"
            )
        max_seq_len = max(seq_lens_host)
        need_prefill_score = True
        compute_query_logits = need_prefill_score and (
            not self.skip_first_token_guard_logits
        )
        self.rec_buf.reset_slots_masked(slot_ids, forward_batch.extend_prefix_lens == 0)
        h0, query_logits, _ = self.cfc_guard.prefill_parallel(
            feat_t,
            seq_lens,
            max_seq_len=max_seq_len,
            seq_lens_host=seq_lens_host,
            compute_query_logits=compute_query_logits,
        )
        self.rec_buf.write_slots(slot_ids, h0)
        if need_prefill_score and (not self.skip_first_token_guard_logits):
            return query_logits
        return None
    h_prev = self.rec_buf.read_slots(slot_ids)
    cfc_input = self.cfc_guard.encode_features(feat_t)
    h_new = self.cfc_guard.gru_step(cfc_input, h_prev)
    self.rec_buf.write_slots(slot_ids, h_new)
    if guard_variant.enabled():
        guard_variant.break_before_guard()
    decode_logits = self.cfc_guard.score(h_new, cfc_input)
    if guard_variant.enabled():
        guard_variant.break_after_guard()
    return decode_logits
    return None


class Glm4vMoePGuardForConditionalGeneration(SglangGlm4vMoeForConditionalGeneration):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.capture_aux_hidden_states = True
        guard_config = config.text_config if hasattr(config, "text_config") else config
        static_layer_indices = capture_layer_indices()
        all_layer_ids = list(range(self.model.start_layer, self.model.end_layer + 1))
        if static_layer_indices:
            capture_layer_ids = [all_layer_ids[1 + idx] for idx in static_layer_indices]
            self.gate_topk = False
        else:
            capture_layer_ids = all_layer_ids[1:]
            self.gate_topk = True
        if self.model.start_layer in capture_layer_ids:
            raise NotImplementedError(
                "[pguard][glm4_5v] capturing layer index 0 is not supported (native capture crashes at residual=None); see glm4v_moe_pguard_stateless._check_capture_layer_ids for the fix hint."
            )
        _final_norm_id = self.model.end_layer
        self._capture_final_norm = _final_norm_id in capture_layer_ids
        self.model.layers_to_capture = [
            i for i in capture_layer_ids if i < _final_norm_id
        ]
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.prefill_risk_mode = PGUARD_PREFILL_RISK_MODE
        if self.prefill_risk_mode not in ("last", "all"):
            raise ValueError(
                f"PGUARD_PREFILL_RISK_MODE must be 'last' or 'all', got {self.prefill_risk_mode!r}"
            )
        self.prefill_score_prob = PGUARD_PREFILL_SCORE_PROB
        _static_layers = capture_layer_indices()
        n_cap = (
            len(_static_layers)
            if _static_layers
            else self.model.end_layer - self.model.start_layer
        )
        validate_cfc_guard_config(
            PGUARD_CFC_STATE_PATH, guard_config.hidden_size, context="GLM cfc checkpoint"
        )
        self.cfc_guard = CfcPluginGuard(
            hidden_size=guard_config.hidden_size,
            n_cap=n_cap,
            proj_dim=PGUARD_CFC_PROJ_DIM,
            mem_dim=PGUARD_CFC_MEM_DIM,
            use_dt=PGUARD_CFC_USE_DT,
            dt=PGUARD_CFC_DT,
            num_classes=PGUARD_CFC_NUM_CLASSES,
            dtype=torch.bfloat16,
            gate_topk=self.gate_topk,
            graph_unroll=PGUARD_CFC_GRAPH_UNROLL,
            use_triton=PGUARD_CFC_USE_TRITON,
            prefill_mode=PGUARD_CFC_PREFILL_MODE,
            prefill_effective_lengths=PGUARD_CFC_PREFILL_EFFECTIVE_LENGTHS,
            classification_heads=resolve_cfc_classification_heads(
                PGUARD_CFC_STATE_PATH
            ),
        )
        self.rec_buf = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        _cfc_state = PGUARD_CFC_STATE_PATH
        if _cfc_state:
            if os.path.exists(_cfc_state):
                load_cfc_guard_state(self.cfc_guard, _cfc_state)
                logger.info(f"[CFC_GUARD] loaded cfc_guard state from {_cfc_state}")
            else:
                logger.warning(
                    f"[CFC_GUARD] PGUARD_CFC_STATE_PATH={_cfc_state} 不存在，cfc_guard 用随机初始化"
                )
        super().load_weights(weights=weights)

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
                    f"multimodal section rotary embedding requires (3, seq_len) positions, but got {positions.size()}"
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
        if self._capture_final_norm and aux_hidden_states is not None:
            aux_hidden_states = list(aux_hidden_states) + [hidden_states]
        if self.pp_group.is_last_rank:
            if not get_embedding:
                guard_hidden_states = []
                if aux_hidden_states is not None:
                    guard_hidden_states.extend(aux_hidden_states)
                result = self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
                guard_logits = _cfc_guard_run(self, guard_hidden_states, forward_batch)
                if guard_logits is not None and guard_variant.enabled():
                    guard_variant.record_eager_guard_execution()
                if guard_logits is not None:
                    assert guard_logits.shape[0] == forward_batch.batch_size, (
                        f"guard_logits.shape: {guard_logits.shape} != {forward_batch.batch_size}"
                    )
                if guard_logits is not None:
                    result.customized_info = {"guard_logits": guard_logits}
                else:
                    result.customized_info = {
                        "guard_logits": [0.0] * forward_batch.batch_size
                    }
                return result
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states


class Glm4vMoeForConditionalGeneration(Glm4vMoePGuardForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
