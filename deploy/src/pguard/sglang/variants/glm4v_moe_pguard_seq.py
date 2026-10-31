import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.glm4v_moe import (
    Glm4vMoeForConditionalGeneration as SglangGlm4vMoeForConditionalGeneration,
)

from ...config import (
    PGUARD_EMA_SCALES,
    PGUARD_HIST_N,
    PGUARD_PREFILL_RISK_MODE,
    PGUARD_PREFILL_SCORE_PROB,
    PGUARD_SEQ_COMPILE,
    PGUARD_SEQ_DECODE_TP_FUSION,
    PGUARD_SEQ_FUSE_CLASSIFIERS,
    PGUARD_SEQ_FUSE_EMA,
    PGUARD_SEQ_FUSE_KERNELS,
    PGUARD_SEQ_NUM_CLASSES,
    PGUARD_SEQ_ROW_SHARD,
    PGUARD_SEQ_STATE_PATH,
    PGUARD_SKIP_FIRST_TOKEN,
    PGUARD_VERSION,
    capture_layer_indices,
    is_debug,
)
from ...guard.history_state import (
    HistoryFeatureBuffer,
    build_score_gather_index,
    shard_score_columns,
)
from ...guard.seq_guard import (
    FusionTpContext,
    SequencePluginGuard,
    load_seq_guard_state,
    require_seq_guard_config,
    resolve_checkpoint_capture_layers,
    resolve_classification_heads,
    resolve_sequence_guard_profile,
    validate_production_classification_schema,
    validate_seq_checkpoint_semantics,
    validate_seq_feature_selection,
    validate_seq_model_identity,
    validate_task_schema,
)
from ...patches import guard_variant
from ...patches.breakable_patch import break_graph

logger = logging.getLogger(__name__)


class _RowShardCtx:

    __slots__ = ("rank", "world", "device_group")

    def __init__(self, rank: int, world: int, device_group) -> None:
        self.rank = rank
        self.world = world
        self.device_group = device_group


def _attn_tp_world(self) -> int:
    if self._attn_world is None:
        from sglang.srt.distributed import (
            get_attn_tensor_model_parallel_world_size,
        )

        self._attn_world = int(get_attn_tensor_model_parallel_world_size())
    return self._attn_world


def _row_shard_ctx(self) -> Optional[_RowShardCtx]:
    if not self.row_shard:
        return None
    if self._shard_ctx is not None:
        return self._shard_ctx
    world = _attn_tp_world(self)
    if world <= 1:
        self.row_shard = False
        logger.info("[SEQ_GUARD] attn_tp world=1，行切分不启用（等价于全量路径）")
        return None
    from sglang.srt.distributed import (
        get_attn_tensor_model_parallel_rank,
        get_attn_tp_group,
    )

    self._shard_ctx = _RowShardCtx(
        rank=get_attn_tensor_model_parallel_rank(),
        world=world,
        device_group=get_attn_tp_group().device_group,
    )
    logger.info(
        f"[SEQ_GUARD] 行切分启用：attn_tp rank={self._shard_ctx.rank}/{world}，"
        f"各 rank 只算 1/{world} 打分行 + all-reduce(MAX) 合并"
    )
    return self._shard_ctx


def _all_reduce_max(x: torch.Tensor, ctx: _RowShardCtx) -> torch.Tensor:
    x = x.contiguous()
    dist.all_reduce(x, op=dist.ReduceOp.MAX, group=ctx.device_group)
    return x


def _decode_tp_fusion_ctx(self) -> Optional[FusionTpContext]:
    if not self.seq_decode_tp_fusion:
        return None
    if self._decode_tp_ctx is not None:
        return self._decode_tp_ctx
    world = _attn_tp_world(self)
    if world <= 1:
        self.seq_decode_tp_fusion = False
        return None
    from sglang.srt.distributed import (
        get_attn_tensor_model_parallel_rank,
        get_attn_tp_group,
    )

    self._decode_tp_ctx = FusionTpContext(
        rank=get_attn_tensor_model_parallel_rank(),
        world=world,
        device_group=get_attn_tp_group().device_group,
    )
    return self._decode_tp_ctx


_SCORE_SEED_BASE = 0x50475541  # "PGUA"


def _assert_sampling_in_sync(checksum: torch.Tensor, ctx: _RowShardCtx) -> None:
    hi = checksum.clone().to(torch.float64)
    lo = hi.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=ctx.device_group)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=ctx.device_group)
    if not torch.equal(hi, lo):
        raise RuntimeError(
            f"[SEQ_GUARD] 各 attn_tp rank 的采样索引不一致（checksum max={hi.item()} "
            f"min={lo.item()}），行切分的 MAX 合并结果不可信。"
            f"检查 build_score_gather_index 的 generator 是否在各 rank 上同种子。"
        )


def _row_shard_blocker() -> Optional[str]:
    from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.moe.utils import (
        should_use_flashinfer_cutlass_moe_fp4_allgather,
    )
    from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled

    if is_prefill_context_parallel_enabled():
        return "CP prefill 已开启，各 rank 的捕获只覆盖自己那片 token"
    if not get_moe_a2a_backend().is_none():
        return f"MoE a2a backend={get_moe_a2a_backend()}，稀疏层输出是 per-rank 分片"
    if should_use_flashinfer_cutlass_moe_fp4_allgather():
        return "flashinfer cutlass MoE fp4 all-gather 使稀疏层输出是 per-rank 分片"
    if enable_moe_dense_fully_dp():
        return "moe_dense_tp_size==1（dense fully DP），稠密层输出是 per-rank 分片"
    return None


def _resolve_max_req(forward_batch: ForwardBatch) -> int:
    pool = getattr(forward_batch, "req_to_token_pool", None)
    if pool is None:
        try:
            from sglang.srt.model_executor.forward_context import get_req_to_token_pool

            pool = get_req_to_token_pool()
        except Exception as e:
            logger.warning(f"[SEQ_GUARD] get_req_to_token_pool 失败: {e}")
    req_to_token = getattr(pool, "req_to_token", None)
    if req_to_token is None or req_to_token.shape[0] <= 0:
        raise RuntimeError(
            "[SEQ_GUARD] 拿不到 req_to_token_pool，拒绝用猜测值分配 hist_buf："
            "slot id 越界会触发 device-side assert。检查当前 sglang 是否提供 "
            "ForwardBatch.req_to_token_pool 或 forward_context.get_req_to_token_pool()。"
        )
    return int(req_to_token.shape[0])


def _ensure_hist_buf(self, forward_batch: ForwardBatch) -> None:
    if self.hist_buf is not None:
        return
    if not self.pp_group.is_last_rank:
        return
    max_req = _resolve_max_req(forward_batch)
    device = self.lm_head.weight.device
    # （RuntimeError: Inplace update to inference tensor outside InferenceMode）。
    with torch.inference_mode(False):
        self.hist_buf = HistoryFeatureBuffer(
            max_req=max_req,
            d_feat=self._seq_d_feat,
            window_n=self.hist_n,
            ema_scales=self.ema_scales,
            dtype=self.lm_head.weight.dtype,
            num_classes=self.sequence_guard.num_classes,
            sampled_scan_chunk=None,
            fused_sampled_ema=PGUARD_SEQ_FUSE_EMA,
        ).to(device)
    logger.info(
        f"[SEQ_GUARD] hist_buf init: max_req={max_req}, D={self._seq_d_feat}, "
        f"N={self.hist_n}, ema_scales={self.ema_scales}, "
        f"num_classes={self.sequence_guard.num_classes}, device={device}"
    )


def _guard_forward(self, hist_seq, hist_mask, ema, feat, prefill: bool):
    if prefill and self.seq_compile:
        if self._guard_compiled is None:
            self._guard_compiled = torch.compile(self.sequence_guard, dynamic=True)
        return self._guard_compiled(hist_seq, hist_mask, ema, feat)
    tp_context = None if prefill else _decode_tp_fusion_ctx(self)
    return self.sequence_guard(
        hist_seq, hist_mask, ema, feat, tp_fusion_context=tp_context
    )


def _seq_guard_run(self, guard_hidden_states, forward_batch: ForwardBatch):
    if not guard_hidden_states:
        return None
    is_decode = forward_batch.forward_mode.is_decode()
    is_extend = forward_batch.forward_mode.is_extend()
    if not (is_decode or is_extend):
        return None
    _ensure_hist_buf(self, forward_batch)
    if self.hist_buf is None:
        return None
    feat_t = torch.cat(list(guard_hidden_states), dim=-1)
    slot_ids = forward_batch.req_pool_indices.to(torch.long)

    if is_extend:
        sample_idx = None
        valid = None
        shard_ctx = None
        shard_checksum = None
        if self.prefill_risk_mode == "all" and self.prefill_score_prob < 1.0:
            shard_ctx = _row_shard_ctx(self)
            world = _attn_tp_world(self)
            self._prefill_counter += 1
            gen = self._score_generator()
            gen.manual_seed(_SCORE_SEED_BASE + self._prefill_counter)
            gather_idx, valid = build_score_gather_index(
                forward_batch.extend_seq_lens.to(torch.long),
                feat_t.size(0),
                self.prefill_score_prob,
                feat_t.device,
                pad_multiple=world,
                generator=gen,
            )
            if shard_ctx is not None:
                if is_debug():
                    shard_checksum = gather_idx.sum()
                gather_idx, valid = shard_score_columns(
                    gather_idx, valid, shard_ctx.rank, shard_ctx.world
                )
            sample_idx = gather_idx.reshape(-1)
        hist_seq, hist_mask, ema = self.hist_buf.prefill_update(
            slot_ids=slot_ids,
            feat_seq=feat_t,
            seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            per_token=(self.prefill_risk_mode == "all"),
            sample_idx=sample_idx,
        )
        if self.prefill_risk_mode == "all":
            seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            B = seq_lens.size(0)
            if sample_idx is not None:
                risk_rows = _guard_forward(
                    self, hist_seq, hist_mask, ema, feat_t[sample_idx], prefill=True
                )  # [B*K_max, num_classes]
                chunk_max = (
                    risk_rows.view(B, -1, risk_rows.size(-1))
                    .masked_fill(~valid.unsqueeze(-1), float("-inf"))
                    .amax(dim=1)
                )  # [B, num_classes]
            else:
                risk_all = _guard_forward(
                    self, hist_seq, hist_mask, ema, feat_t, prefill=True
                )  # [T, num_classes]
                prompt_idx = torch.repeat_interleave(
                    torch.arange(B, device=feat_t.device),
                    seq_lens,
                    output_size=feat_t.size(0),
                )
                chunk_max = torch.full(
                    (B, risk_all.size(-1)),
                    float("-inf"),
                    device=feat_t.device,
                    dtype=risk_all.dtype,
                )
                chunk_max.scatter_reduce_(
                    0,
                    prompt_idx.unsqueeze(-1).expand(-1, risk_all.size(-1)),
                    risk_all,
                    reduce="amax",
                    include_self=True,
                )
            if shard_ctx is not None:
                chunk_max = _all_reduce_max(chunk_max, shard_ctx)
                if shard_checksum is not None:
                    _assert_sampling_in_sync(shard_checksum, shard_ctx)
            return self.hist_buf.update_risk_max(slot_ids, chunk_max)
        if not self.skip_first_token_guard_logits:
            seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            last_indices = torch.cumsum(seq_lens, dim=0) - 1
            return _guard_forward(
                self, hist_seq, hist_mask, ema, feat_t[last_indices], prefill=True
            )
        return None

    hist_seq, hist_mask, ema = self.hist_buf.read_and_update(
        slot_ids=slot_ids,
        feat_seq=feat_t,
        is_extend=False,
        extend_prefix_lens=None,
    )
    if guard_variant.enabled():
        guard_variant.break_before_guard()
    guard_logits = _guard_forward(
        self, hist_seq, hist_mask, ema, feat_t, prefill=False
    )
    if guard_variant.enabled():
        break_graph()
    return guard_logits


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
        guard_config = config.text_config if hasattr(config, "text_config") else config
        if server_args.enable_two_batch_overlap:
            raise NotImplementedError(
                "PGuard aux capture is incompatible with --enable-two-batch-overlap "
                "on GLM-4.5V: Glm4MoeModel routes layers after first_k_dense_replace "
                "through model_forward_maybe_tbo, which does not capture hidden states"
            )
        self.gate_topk = self._setup_capture_layers()
        self.skip_first_token_guard_logits = PGUARD_SKIP_FIRST_TOKEN
        self.hist_n = PGUARD_HIST_N
        self.ema_scales = PGUARD_EMA_SCALES
        self.prefill_risk_mode = PGUARD_PREFILL_RISK_MODE
        if self.prefill_risk_mode not in ("last", "all"):
            raise ValueError(
                f"PGUARD_PREFILL_RISK_MODE must be 'last' or 'all', "
                f"got {self.prefill_risk_mode!r}"
            )
        self.prefill_score_prob = PGUARD_PREFILL_SCORE_PROB
        self.seq_fuse_kernels = PGUARD_SEQ_FUSE_KERNELS
        self.seq_compile = PGUARD_SEQ_COMPILE
        self.seq_decode_tp_fusion = PGUARD_SEQ_DECODE_TP_FUSION
        self._decode_tp_ctx = None
        self._guard_compiled = None

        self.row_shard = PGUARD_SEQ_ROW_SHARD
        if self.row_shard or self.seq_decode_tp_fusion:
            blocker = _row_shard_blocker()
            if blocker is not None:
                if self.row_shard:
                    self.row_shard = False
                    logger.warning(
                        f"[SEQ_GUARD] PGUARD_SEQ_ROW_SHARD=1 但前提不成立，"
                        f"已退回全量路径：{blocker}"
                    )
                if self.seq_decode_tp_fusion:
                    self.seq_decode_tp_fusion = False
                    logger.warning(
                        f"[SEQ_GUARD] PGUARD_SEQ_DECODE_TP_FUSION=1 但前提不成立，"
                        f"已退回完整权重路径：{blocker}"
                    )
        self._shard_ctx = None
        self._attn_world = None
        self._prefill_counter = 0
        self._score_gen = None

        static_layers = tuple(capture_layer_indices())
        n_cap = (
            len(static_layers)
            if static_layers
            else (self.model.end_layer - self.model.start_layer - 1)
        )
        self._seq_d_feat = n_cap * guard_config.hidden_size
        head_config = require_seq_guard_config(
            PGUARD_SEQ_STATE_PATH, "GLM sequence checkpoint"
        )
        profile = resolve_sequence_guard_profile(
            head_config,
            explicit_version=PGUARD_VERSION,
        )
        if head_config.get("pguard_version"):
            classification_heads = validate_production_classification_schema(
                head_config, context="GLM sequence checkpoint"
            )
            validate_seq_model_identity(
                head_config,
                ("glm4v_moe",),
                context="GLM sequence checkpoint",
                base_model_path=server_args.model_path,
            )
            validate_seq_feature_selection(
                head_config,
                tuple(layer + 1 for layer in static_layers),
                context="GLM sequence checkpoint",
            )
        else:
            classification_heads = resolve_classification_heads(head_config)
            if classification_heads:
                validate_task_schema(
                    head_config,
                    classification_heads,
                    context="GLM legacy sequence checkpoint",
                )
        checkpoint_capture = resolve_checkpoint_capture_layers(
            head_config, selected_layer_offset=-1
        )
        if checkpoint_capture and checkpoint_capture != static_layers:
            raise ValueError(
                f"checkpoint capture layers {checkpoint_capture} != "
                f"PGUARD_CAPTURE_LAYERS {static_layers}"
            )
        self._seq_checkpoint_config = head_config
        self._seq_semantics = {
            "pguard_version": profile.version,
            "d_feat": self._seq_d_feat,
            "hidden_size": guard_config.hidden_size,
            "n_cap": n_cap,
            "window_n": self.hist_n,
            "ema_update_mask": "attention_mask",
            "ema_scales": self.ema_scales,
            "n_bert_layers": profile.n_bert_layers,
            "n_adapter_layers": profile.n_adapter_layers,
            "n_attn_heads": 8,
            "top_L": 3,
            "gate_first": profile.gate_first,
            "num_classes": (
                sum(size for _, size in classification_heads)
                if classification_heads
                else PGUARD_SEQ_NUM_CLASSES
            ),
        }
        validate_seq_checkpoint_semantics(
            head_config,
            None,
            self._seq_semantics,
            context="GLM sequence checkpoint",
            require_formal_config_metadata=True,
        )
        self.sequence_guard_profile = profile
        self.sequence_guard = SequencePluginGuard(
            d_feat=self._seq_d_feat,
            window_n=self.hist_n,
            n_cap=n_cap,
            n_scales=len(self.ema_scales),
            n_bert_layers=profile.n_bert_layers,
            n_adapter_layers=profile.n_adapter_layers,
            num_classes=PGUARD_SEQ_NUM_CLASSES,
            dtype=torch.bfloat16,
            gate_topk=self.gate_topk,
            fuse_kernels=self.seq_fuse_kernels,
            fuse_classifiers=PGUARD_SEQ_FUSE_CLASSIFIERS,
            classification_heads=classification_heads,
        )
        logger.info(
            f"[SEQ_GUARD] profile={profile.pguard_version} gate_first={profile.gate_first} "
            f"fusion_layers={profile.n_bert_layers} adapter_layers={profile.n_adapter_layers}"
        )
        self.hist_buf = None

    def _score_generator(self) -> torch.Generator:
        if self._score_gen is None:
            self._score_gen = torch.Generator()  # CPU
        return self._score_gen

    def _setup_capture_layers(self) -> bool:
        """Configure global GLM layer-entry captures only on the last PP rank."""
        self.capture_aux_hidden_states = False
        static_layer_indices = tuple(capture_layer_indices())
        if not self.pp_group.is_last_rank:
            return not static_layer_indices
        if not static_layer_indices:
            capture_layer_ids = list(
                range(max(1, self.model.start_layer), self.model.end_layer)
            )
            self.model.layers_to_capture = capture_layer_ids
            self.capture_aux_hidden_states = True
            return True
        capture_layer_ids = [layer + 1 for layer in static_layer_indices]
        missing = [
            layer
            for layer in capture_layer_ids
            if not self.model.start_layer <= layer < self.model.end_layer
        ]
        if missing:
            raise ValueError(
                "all PGuard capture layers must live on the last PP rank; "
                f"missing={missing}, last-rank range="
                f"[{self.model.start_layer}, {self.model.end_layer})"
            )
        self.model.layers_to_capture = capture_layer_ids
        self.capture_aux_hidden_states = True
        return False

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        if PGUARD_SEQ_STATE_PATH:
            if os.path.exists(PGUARD_SEQ_STATE_PATH):
                meta = load_seq_guard_state(
                    self.sequence_guard, PGUARD_SEQ_STATE_PATH, strict=True
                )
                validate_seq_checkpoint_semantics(
                    self._seq_checkpoint_config,
                    meta,
                    self._seq_semantics,
                    context="GLM sequence checkpoint",
                    require_formal_config_metadata=True,
                )
                logger.info(
                    f"[SEQ_GUARD] loaded sequence_guard state from "
                    f"{PGUARD_SEQ_STATE_PATH}, meta={meta}"
                )
            else:
                raise FileNotFoundError(
                    f"PGUARD_SEQ_STATE_PATH does not exist: {PGUARD_SEQ_STATE_PATH}"
                )
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

        result = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
        guard_logits = _seq_guard_run(self, aux_hidden_states, forward_batch)
        if guard_logits is not None and guard_variant.enabled():
            guard_variant.record_eager_guard_execution()
        if guard_logits is None:
            if is_debug():
                logger.error(
                    f"guard_logits is None, "
                    f"is_aux_hidden_states_none={aux_hidden_states is None}, "
                    f"{forward_batch.batch_size=}"
                )
            result.customized_info = {"guard_logits": [0.0] * forward_batch.batch_size}
            return result
        assert guard_logits.shape[0] == forward_batch.batch_size, (
            f"guard_logits.shape: {guard_logits.shape} != {forward_batch.batch_size}"
        )
        if is_debug():
            logger.warning(f"{guard_logits.shape=}")
        result.customized_info = {"guard_logits": guard_logits}
        return result


class Glm4vMoeForConditionalGeneration(Glm4vMoePGuardForConditionalGeneration):
    pass


