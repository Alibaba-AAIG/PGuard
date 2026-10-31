import logging
import os
from typing import Iterable, Optional, Tuple
import torch
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.utils.cp_utils import prepare_context_parallel_metadata
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM as SglangDeepseekV4ForCausalLM,
)
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
    PGUARD_CFC_PREFILL_TP_DEDUP,
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
    validate_cfc_prefill_host_metadata,
)
from .cfc_prefill_tp import resolve_cfc_prefill_tp_context, run_cfc_prefill_tp_dedup

logger = logging.getLogger(__name__)


def _resolve_dspark_layer_ids(static_layer_indices, all_layer_ids):
    if not static_layer_indices:
        return all_layer_ids
    max_index = max(static_layer_indices)
    if min(static_layer_indices) < 0 or max_index >= len(all_layer_ids):
        raise ValueError(
            f"PGUARD_CAPTURE_LAYERS={static_layer_indices} 超出可捕获范围 [0, {len(all_layer_ids) - 1}]"
        )
    return [all_layer_ids[index] for index in static_layer_indices]


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
    feat_t = torch.stack(guard_hidden_states, dim=1)
    slot_ids = forward_batch.req_pool_indices.to(torch.long)
    if is_extend:
        seq_lens = forward_batch.extend_seq_lens.to(torch.long)
        seq_lens_host = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_lens = forward_batch.extend_prefix_lens
        max_seq_len = validate_cfc_prefill_host_metadata(
            seq_lens_host,
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            batch_size=forward_batch.batch_size,
            total_tokens=feat_t.size(0),
        )
        if prefix_lens is None:
            raise RuntimeError(
                "CFC guard does not support radix-prefix reuse because the recurrent prefix state is unavailable"
            )
        self.rec_buf.reset_slots_masked(slot_ids, prefix_lens == 0)
        if self.cfc_prefill_tp_dedup:
            if self._cfc_prefill_tp_context is None:
                self._cfc_prefill_tp_context = resolve_cfc_prefill_tp_context()
                logger.info(
                    "[CFC_GUARD] prefill TP dedup enabled: attn_tp rank=%d/%d",
                    self._cfc_prefill_tp_context.rank,
                    self._cfc_prefill_tp_context.world,
                )
            h0, query_logits, _ = run_cfc_prefill_tp_dedup(
                self.cfc_guard,
                feat_t,
                seq_lens,
                self._cfc_prefill_tp_context,
                max_seq_len=max_seq_len,
                seq_lens_host=seq_lens_host,
            )
        else:
            h0, query_logits, _ = self.cfc_guard.prefill_parallel(
                feat_t, seq_lens, max_seq_len=max_seq_len, seq_lens_host=seq_lens_host
            )
        self.rec_buf.write_slots(slot_ids, h0)
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


class DeepseekV4PGuardForCausalLM(SglangDeepseekV4ForCausalLM):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        main_freqs_cis = None
        for layer in self.model.layers:
            attention = layer.self_attn
            if attention.compress_ratio == 0:
                if main_freqs_cis is None:
                    main_freqs_cis = precompute_freqs_cis(
                        dim=attention.qk_rope_head_dim,
                        seqlen=config.max_position_embeddings,
                        original_seq_len=0,
                        base=attention.rope_base,
                        factor=1.0,
                        beta_fast=32,
                        beta_slow=1,
                    ).to(attention.freqs_cis.device)
                attention.freqs_cis = main_freqs_cis
        self._capture_layer_indices = tuple(capture_layer_indices())
        self._resolved_dspark_layer_ids = ()
        self._resolved_hf_hidden_state_ids = ()
        all_layer_ids = list(
            range(self.model.start_layer, self.model.end_layer - 1)
        ) + [self.model.end_layer]
        capture_layer_ids = _resolve_dspark_layer_ids(
            self._capture_layer_indices, all_layer_ids
        )
        self.gate_topk = not self._capture_layer_indices
        self._resolved_dspark_layer_ids = tuple(capture_layer_ids)
        self._resolved_hf_hidden_state_ids = tuple(
            (
                layer_id + 1
                if layer_id < self.model.end_layer
                else self.model.end_layer
                for layer_id in capture_layer_ids
            )
        )
        _final_norm_id = self.model.end_layer
        self._capture_final_norm = _final_norm_id in capture_layer_ids
        self.set_dspark_layers_to_capture(
            [i for i in capture_layer_ids if i < _final_norm_id]
        )
        logger.info(
            "[CFC_GUARD] capture indices=%s resolved dspark ids=%s final_norm=%s gate_topk=%s",
            self._capture_layer_indices,
            self._resolved_dspark_layer_ids,
            self._capture_final_norm,
            self.gate_topk,
        )
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.prefill_risk_mode = PGUARD_PREFILL_RISK_MODE
        if self.prefill_risk_mode not in ("last", "all"):
            raise ValueError(
                f"PGUARD_PREFILL_RISK_MODE must be 'last' or 'all', got {self.prefill_risk_mode!r}"
            )
        self.prefill_score_prob = PGUARD_PREFILL_SCORE_PROB
        self.cfc_prefill_tp_dedup = PGUARD_CFC_PREFILL_TP_DEDUP
        if self.cfc_prefill_tp_dedup and self.dsa_enable_prefill_cp:
            self.cfc_prefill_tp_dedup = False
            logger.warning(
                "[CFC_GUARD] PGUARD_CFC_PREFILL_TP_DEDUP=1 but CP prefill is enabled; TP dedup is disabled because captured features are not replicated"
            )
        self._cfc_prefill_tp_context = None
        n_cap = (
            len(self._capture_layer_indices)
            or self.model.end_layer - self.model.start_layer
        )
        validate_cfc_guard_config(
            PGUARD_CFC_STATE_PATH, config.hidden_size, context="DeepSeek cfc checkpoint"
        )
        self.cfc_guard = CfcPluginGuard(
            hidden_size=config.hidden_size,
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
        if _cfc_state and os.path.exists(_cfc_state):
            meta = load_cfc_guard_state(self.cfc_guard, _cfc_state, strict=False)
            logger.info(
                "[CFC_GUARD] loaded cfc_guard state from %s, meta=%s", _cfc_state, meta
            )
        else:
            logger.warning(
                "[CFC_GUARD] PGUARD_CFC_STATE_PATH=%r 不存在，cfc_guard 用随机初始化",
                _cfc_state,
            )
        super().load_weights(weights=weights)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states
        if self._capture_final_norm and aux_hidden_states is not None:
            aux_hidden_states = list(aux_hidden_states) + [hidden_states]
        guard_hidden_states = []
        if aux_hidden_states is not None:
            guard_hidden_states.extend(aux_hidden_states)
        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=None
            if aux_hidden_states is not None
            else pre_hc_head,
        )
        guard_logits = _cfc_guard_run(
            self, guard_hidden_states, forward_batch, input_ids, positions
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
            result.customized_info = {"guard_logits": [0.0] * forward_batch.batch_size}
        return result


class DeepseekV4ForCausalLM(DeepseekV4PGuardForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
