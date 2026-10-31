import logging
import os
from typing import Iterable, Optional, Tuple
import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as SglangQwen3_5ForConditionalGeneration,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGeneration as SglangQwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
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
    PGUARD_CFC_HF_ALIGNED_BACKBONE,
    PGUARD_SKIP_FIRST_TOKEN,
    PGUARD_CFC_COMPILE,
    PGUARD_CFC_STACK_BUFFER,
    capture_layer_indices,
)
from ...patches import guard_variant
from ...guard.recurrent_state import RecurrentStateBuffer
from ...guard.cfc_guard import (
    CfcPluginGuard,
    load_cfc_guard_state,
    resolve_cfc_classification_heads,
    validate_cfc_guard_config,
    validate_cfc_prefill_host_metadata,
)

logger = logging.getLogger(__name__)


def _cfc_capture_layer_ids(config, all_layer_ids, static_layer_indices):
    if config.get("pguard_capture_layers") is None:
        raise ValueError("CFC config pguard_capture_layers is required")
    if static_layer_indices:
        capture_ids = tuple(static_layer_indices)
    else:
        capture_ids = tuple(int(item) for item in config["pguard_capture_layers"].split(","))
    missing = [layer for layer in capture_ids if layer not in all_layer_ids]
    if missing:
        raise ValueError(f"CFC capture layers are outside the local model range: {missing}")
    return capture_ids


def _stack_guard_hidden(guard_hidden_states, buf_cache):
    if not guard_hidden_states:
        return None
    if not PGUARD_CFC_STACK_BUFFER:
        return torch.stack(guard_hidden_states, dim=1)
    n_cap = len(guard_hidden_states)
    tokens = guard_hidden_states[0].shape[0]
    hidden = guard_hidden_states[0].shape[1]
    device = guard_hidden_states[0].device
    dtype = guard_hidden_states[0].dtype
    key = (n_cap, hidden, device, dtype)
    buf = buf_cache.get(key)
    if buf is None or buf.shape[0] < tokens:
        alloc_size = max(tokens, 8192)
        buf = torch.empty(alloc_size, n_cap, hidden, device=device, dtype=dtype)
        buf_cache[key] = buf
    out = buf[:tokens]
    for i, h in enumerate(guard_hidden_states):
        out[:, i, :].copy_(h)
    return out


class _CfcPrefillWrapper(torch.nn.Module):
    def __init__(self, cfc_guard):
        super().__init__()
        self.cfc_guard = cfc_guard

    def forward(self, feat_t, seq_lens, max_seq_len, compute_query_logits):
        return self.cfc_guard.prefill_parallel(
            feat_t,
            seq_lens,
            max_seq_len=max_seq_len,
            compute_query_logits=compute_query_logits,
        )


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
    try:
        from sglang.srt.server_args import get_global_server_args

        _mrr = getattr(get_global_server_args(), "max_running_requests", None)
        if isinstance(_mrr, int) and _mrr > 0:
            max_req = max(max_req, _mrr + 1)
    except Exception:
        pass
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
    if not hasattr(self, "_stack_buf_cache"):
        self._stack_buf_cache = {}
    if PGUARD_CFC_COMPILE and (not hasattr(self, "_prefill_wrapper")):
        self._prefill_wrapper = _CfcPrefillWrapper(self.cfc_guard)
        self._prefill_compiled = None


def _cfc_guard_run(self, guard_hidden_states, forward_batch, input_ids, positions):
    if not guard_hidden_states:
        return None
    fm = forward_batch.forward_mode
    is_decode = fm.is_decode()
    is_split_prefill = hasattr(fm, "is_split_prefill") and fm.is_split_prefill()
    is_extend = fm.is_extend() and (not is_split_prefill)
    if is_split_prefill:
        raise RuntimeError(
            "CFC guard does not support split prefill because the recurrent prefix state is unavailable"
        )
    if not (is_decode or is_extend):
        return None
    _cfc_guard_ensure_rec_buf(self, forward_batch)
    if self.rec_buf is None:
        return None
    feat_t = _stack_guard_hidden(guard_hidden_states, self._stack_buf_cache)
    slot_ids = forward_batch.req_pool_indices.to(torch.long)
    if is_extend:
        seq_lens = forward_batch.extend_seq_lens.to(torch.long)
        seq_lens_host = getattr(forward_batch, "extend_seq_lens_cpu", None)
        real_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if getattr(forward_batch, "extend_num_tokens", None) != real_tokens:
            raise ValueError(
                f"CFC prefill real-token metadata disagrees: extend_num_tokens={getattr(forward_batch, 'extend_num_tokens', None)}, num_token_non_padded_cpu={real_tokens}"
            )
        max_seq_len = validate_cfc_prefill_host_metadata(
            seq_lens_host,
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            batch_size=forward_batch.batch_size,
            total_tokens=real_tokens,
            feature_tokens=feat_t.size(0),
        )
        feat_t = feat_t[:real_tokens]
        prefix_lens = forward_batch.extend_prefix_lens
        if prefix_lens is None:
            raise RuntimeError(
                "CFC guard does not support radix-prefix reuse because the recurrent prefix state is unavailable"
            )
        need_prefill_score = True
        compute_query_logits = need_prefill_score and (
            not self.skip_first_token_guard_logits
        )
        self.rec_buf.reset_slots_masked(slot_ids, prefix_lens == 0)
        if (
            PGUARD_CFC_COMPILE
            and (not getattr(self.cfc_guard, "loop_unpack", False))
            and hasattr(self, "_prefill_wrapper")
        ):
            if self._prefill_compiled is None:
                self._prefill_compiled = torch.compile(
                    self._prefill_wrapper, dynamic=True
                )
            h0, query_logits, _ = self._prefill_compiled(
                feat_t, seq_lens, max_seq_len, compute_query_logits
            )
        else:
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


class Qwen3_5PGuardForConditionalGeneration(SglangQwen3_5ForConditionalGeneration):
    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ):
        super().__init__(config, quant_config, prefix, language_model_cls)
        self.capture_aux_hidden_states = True
        guard_config = config.text_config if hasattr(config, "text_config") else config
        static_layer_indices = capture_layer_indices()
        all_layer_ids = list(range(self.model.start_layer, self.model.end_layer))
        self.cfc_guard_config = validate_cfc_guard_config(
            PGUARD_CFC_STATE_PATH, guard_config.hidden_size, context="Qwen cfc checkpoint"
        )
        capture_layer_ids = _cfc_capture_layer_ids(
            self.cfc_guard_config, all_layer_ids, static_layer_indices
        )
        self.model.set_dflash_layers_to_capture(list(capture_layer_ids))
        self.gate_topk = False
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.prefill_risk_mode = PGUARD_PREFILL_RISK_MODE
        if self.prefill_risk_mode not in ("last", "all"):
            raise ValueError(
                f"PGUARD_PREFILL_RISK_MODE must be 'last' or 'all', got {self.prefill_risk_mode!r}"
            )
        self.prefill_score_prob = PGUARD_PREFILL_SCORE_PROB
        n_cap = len(capture_layer_ids)
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _cfc_state = PGUARD_CFC_STATE_PATH
        if _cfc_state:
            if os.path.exists(_cfc_state):
                load_cfc_guard_state(
                    self.cfc_guard,
                    _cfc_state,
                    strict=True,
                )
                logger.info(f"[CFC_GUARD] loaded cfc_guard state from {_cfc_state}")
            else:
                logger.warning(
                    f"[CFC_GUARD] PGUARD_CFC_STATE_PATH={_cfc_state} 不存在，cfc_guard 用随机初始化"
                )
        super().load_weights(weights=weights)
        if PGUARD_CFC_HF_ALIGNED_BACKBONE:
            from ...patches.cfc_hf_backbone import install_cfc_hf_aligned_backbone

            install_cfc_hf_aligned_backbone(self.model)

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
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states
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
                guard_logits = _cfc_guard_run(
                    self,
                    guard_hidden_states,
                    forward_batch,
                    input_ids,
                    forward_batch.positions,
                )
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


class Qwen3_5PGuardMoeForConditionalGeneration(
    SglangQwen3_5MoeForConditionalGeneration
):
    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        self.capture_aux_hidden_states = True
        guard_config = config.text_config if hasattr(config, "text_config") else config
        static_layer_indices = capture_layer_indices()
        all_layer_ids = list(range(self.model.start_layer, self.model.end_layer))
        self.cfc_guard_config = validate_cfc_guard_config(
            PGUARD_CFC_STATE_PATH, guard_config.hidden_size, context="Qwen cfc checkpoint"
        )
        capture_layer_ids = _cfc_capture_layer_ids(
            self.cfc_guard_config, all_layer_ids, static_layer_indices
        )
        self.model.set_dflash_layers_to_capture(list(capture_layer_ids))
        self.gate_topk = False
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.prefill_risk_mode = PGUARD_PREFILL_RISK_MODE
        if self.prefill_risk_mode not in ("last", "all"):
            raise ValueError(
                f"PGUARD_PREFILL_RISK_MODE must be 'last' or 'all', got {self.prefill_risk_mode!r}"
            )
        self.prefill_score_prob = PGUARD_PREFILL_SCORE_PROB
        n_cap = len(capture_layer_ids)
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _cfc_state = PGUARD_CFC_STATE_PATH
        if _cfc_state:
            if os.path.exists(_cfc_state):
                load_cfc_guard_state(
                    self.cfc_guard,
                    _cfc_state,
                    strict=True,
                )
                logger.info(f"[CFC_GUARD] loaded cfc_guard state from {_cfc_state}")
            else:
                logger.warning(
                    f"[CFC_GUARD] PGUARD_CFC_STATE_PATH={_cfc_state} 不存在，cfc_guard 用随机初始化"
                )
        super().load_weights(weights=weights)
        if PGUARD_CFC_HF_ALIGNED_BACKBONE:
            from ...patches.cfc_hf_backbone import install_cfc_hf_aligned_backbone

            install_cfc_hf_aligned_backbone(self.model)

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
            use_deepstack=self.use_deepstack,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states
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
                guard_logits = _cfc_guard_run(
                    self,
                    guard_hidden_states,
                    forward_batch,
                    input_ids,
                    forward_batch.positions,
                )
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


class Qwen3_5ForConditionalGeneration(Qwen3_5PGuardForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5PGuardMoeForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
