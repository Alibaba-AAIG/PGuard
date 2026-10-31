import logging
import os
import weakref
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models import deepseek_v4 as _dsv4
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM as SglangDeepseekV4ForCausalLM,
)

from ...config import (
    PGUARD_EMA_SCALES,
    PGUARD_HIST_N,
    PGUARD_PREFILL_RISK_MODE,
    PGUARD_PREFILL_SCORE_PROB,
    PGUARD_SEQ_COMPILE,
    PGUARD_SEQ_DECODE_TP_FUSION,
    PGUARD_SEQ_FUSED_SAMPLED_EMA,
    PGUARD_SEQ_FUSE_CLASSIFIERS,
    PGUARD_SEQ_FUSE_KERNELS,
    PGUARD_SEQ_NUM_CLASSES,
    PGUARD_SEQ_PREFILL_OVERLAP,
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
    resolve_checkpoint_capture_layers,
    resolve_classification_heads,
    require_seq_guard_config,
    resolve_seq_guard_profile,
    validate_production_classification_schema,
    validate_seq_checkpoint_semantics,
    validate_seq_feature_selection,
    validate_seq_model_identity,
    validate_task_schema,
)
from ...patches import guard_variant
from ...patches.breakable_patch import break_graph

logger = logging.getLogger(__name__)


class _PreparedSampledPrefill:
    __slots__ = (
        "feat_t",
        "slot_ids",
        "sample_idx",
        "valid",
        "shard_ctx",
        "shard_checksum",
        "hist_seq",
        "hist_mask",
        "ema",
    )

    def __init__(
        self,
        feat_t,
        slot_ids,
        sample_idx,
        valid,
        shard_ctx,
        shard_checksum,
        hist_seq,
        hist_mask,
        ema,
    ):
        self.feat_t = feat_t
        self.slot_ids = slot_ids
        self.sample_idx = sample_idx
        self.valid = valid
        self.shard_ctx = shard_ctx
        self.shard_checksum = shard_checksum
        self.hist_seq = hist_seq
        self.hist_mask = hist_mask
        self.ema = ema


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
        logger.info("[SEQ_GUARD] decode fusion TP world=1，自动退回完整权重路径")
        return None
    from sglang.srt.distributed import (
        get_attn_tensor_model_parallel_rank,
        get_attn_tp_group,
    )

    group = get_attn_tp_group()
    self._decode_tp_ctx = FusionTpContext(
        rank=get_attn_tensor_model_parallel_rank(),
        world=world,
        device_group=group.device_group,
    )
    logger.info(
        f"[SEQ_GUARD] decode fusion TP 启用：rank="
        f"{self._decode_tp_ctx.rank}/{world}，每层 2 次 SUM all-reduce"
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
            fused_sampled_ema=self.fused_sampled_ema,
        ).to(device)
    logger.info(
        f"[SEQ_GUARD] hist_buf init: max_req={max_req}, D={self._seq_d_feat}, "
        f"N={self.hist_n}, ema_scales={self.ema_scales}, "
        f"num_classes={self.sequence_guard.num_classes}, "
        f"fused_sampled_ema={self.fused_sampled_ema}, device={device}"
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


def _prepare_sampled_prefill(
    self,
    feat_t: torch.Tensor,
    slot_ids: torch.Tensor,
    forward_batch: ForwardBatch,
) -> _PreparedSampledPrefill:
    shard_ctx = _row_shard_ctx(self)
    world = _attn_tp_world(self)
    self._prefill_counter += 1
    generator = self._score_generator()
    generator.manual_seed(_SCORE_SEED_BASE + self._prefill_counter)
    gather_idx, valid = build_score_gather_index(
        forward_batch.extend_seq_lens.to(torch.long),
        feat_t.size(0),
        self.prefill_score_prob,
        feat_t.device,
        pad_multiple=world,
        generator=generator,
    )
    shard_checksum = None
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
        per_token=True,
        sample_idx=sample_idx,
    )
    return _PreparedSampledPrefill(
        feat_t,
        slot_ids,
        sample_idx,
        valid,
        shard_ctx,
        shard_checksum,
        hist_seq,
        hist_mask,
        ema,
    )


def _finish_sampled_prefill(
    self,
    prepared: _PreparedSampledPrefill,
    forward_batch: ForwardBatch,
):
    seq_lens = forward_batch.extend_seq_lens.to(torch.long)
    batch_size = seq_lens.size(0)
    risk_rows = _guard_forward(
        self,
        prepared.hist_seq,
        prepared.hist_mask,
        prepared.ema,
        prepared.feat_t[prepared.sample_idx],
        prefill=True,
    )
    chunk_max = (
        risk_rows.view(batch_size, -1, risk_rows.size(-1))
        .masked_fill(~prepared.valid.unsqueeze(-1), float("-inf"))
        .amax(dim=1)
    )
    if prepared.shard_ctx is not None:
        chunk_max = _all_reduce_max(chunk_max, prepared.shard_ctx)
        if prepared.shard_checksum is not None:
            _assert_sampling_in_sync(
                prepared.shard_checksum, prepared.shard_ctx
            )
    return self.hist_buf.update_risk_max(prepared.slot_ids, chunk_max)


def _record_tensors_on_stream(values, stream: torch.cuda.Stream) -> None:
    for value in values:
        if torch.is_tensor(value):
            value.record_stream(stream)


def _prefill_overlap_eligible(self, forward_batch: ForwardBatch) -> bool:
    return (
        self.seq_prefill_overlap
        and self.pp_group.is_last_rank
        and forward_batch.forward_mode.is_extend_without_speculative()
        and not forward_batch.forward_mode.is_cuda_graph()
        and self.prefill_risk_mode == "all"
        and self.prefill_score_prob < 1.0
        and bool(self._capture_layer_ids)
        and self._capture_layer_ids[-1] < self.model.end_layer - 1
    )


def _prefill_overlap_hook(self, layer_id: int, layer, output) -> None:
    forward_batch = self._prefill_overlap_forward_batch
    if forward_batch is None:
        return
    expected_index = len(self._prefill_overlap_features)
    if (
        expected_index >= len(self._capture_layer_ids)
        or self._capture_layer_ids[expected_index] != layer_id
    ):
        raise RuntimeError(
            f"[SEQ_GUARD] J capture order mismatch at layer {layer_id}: "
            f"expected {self._capture_layer_ids[expected_index:expected_index + 1]}"
        )
    hidden_states, residual, post, comb = output
    if residual is not None and layer.use_fused_mhc_post_pre:
        completed = layer.hc_post(hidden_states, residual, post, comb)
    else:
        completed = hidden_states
    feature = completed.mean(dim=1)
    self._prefill_overlap_features.append(feature)
    if layer_id != self._capture_layer_ids[-1]:
        return

    main_stream = torch.cuda.current_stream(feature.device)
    if self._prefill_overlap_stream is None:
        self._prefill_overlap_stream = torch.cuda.Stream(device=feature.device)
    side_stream = self._prefill_overlap_stream
    ready = torch.cuda.Event()
    done = torch.cuda.Event()
    ready.record(main_stream)
    _record_tensors_on_stream(self._prefill_overlap_features, side_stream)
    _record_tensors_on_stream(
        (
            forward_batch.req_pool_indices,
            forward_batch.extend_seq_lens,
            forward_batch.extend_prefix_lens,
        ),
        side_stream,
    )
    for buffer in self.hist_buf.buffers():
        buffer.record_stream(side_stream)
    with torch.cuda.stream(side_stream):
        side_stream.wait_event(ready)
        feat_t = torch.cat(self._prefill_overlap_features, dim=-1)
        slot_ids = forward_batch.req_pool_indices.to(torch.long)
        prepared = _prepare_sampled_prefill(
            self, feat_t, slot_ids, forward_batch
        )
        done.record(side_stream)
    self._prefill_overlap_prepared = prepared
    self._prefill_overlap_done = done


def _begin_prefill_overlap(self, forward_batch: ForwardBatch):
    if not _prefill_overlap_eligible(self, forward_batch):
        return None
    _ensure_hist_buf(self, forward_batch)
    if self.hist_buf is None:
        raise RuntimeError("[SEQ_GUARD] J requires hist_buf on the last PP rank")
    self._prefill_overlap_forward_batch = forward_batch
    self._prefill_overlap_features = []
    self._prefill_overlap_prepared = None
    self._prefill_overlap_done = None
    capture_layers = self.model.dspark_layers_to_capture
    self.model.dspark_layers_to_capture = []
    return capture_layers


def _restore_prefill_capture(self, capture_layers) -> None:
    if capture_layers is None:
        return
    self.model.dspark_layers_to_capture = capture_layers
    self._prefill_overlap_forward_batch = None


def _consume_prefill_overlap(self, forward_batch: ForwardBatch):
    prepared = self._prefill_overlap_prepared
    done = self._prefill_overlap_done
    if prepared is None or done is None:
        raise RuntimeError(
            "[SEQ_GUARD] J enabled but the last capture-layer hook did not prepare prefill state"
        )
    main_stream = torch.cuda.current_stream(prepared.feat_t.device)
    main_stream.wait_event(done)
    _record_tensors_on_stream(
        (
            prepared.feat_t,
            prepared.slot_ids,
            prepared.sample_idx,
            prepared.valid,
            prepared.shard_checksum,
            prepared.hist_seq,
            prepared.hist_mask,
            prepared.ema,
        ),
        main_stream,
    )
    for buffer in self.hist_buf.buffers():
        buffer.record_stream(main_stream)
    try:
        return _finish_sampled_prefill(self, prepared, forward_batch)
    finally:
        self._prefill_overlap_features = []
        self._prefill_overlap_prepared = None
        self._prefill_overlap_done = None


def _discard_prefill_overlap(self) -> None:
    done = self._prefill_overlap_done
    if done is not None:
        main_stream = torch.cuda.current_stream()
        main_stream.wait_event(done)
        for buffer in self.hist_buf.buffers():
            buffer.record_stream(main_stream)
    self._prefill_overlap_forward_batch = None
    self._prefill_overlap_features = []
    self._prefill_overlap_prepared = None
    self._prefill_overlap_done = None


def _install_prefill_overlap_hooks(self) -> None:
    self._prefill_overlap_hook_handles = []
    if not self.seq_prefill_overlap:
        return
    if not self._capture_layer_ids:
        logger.warning(
            "[SEQ_GUARD] J requested without static capture layers; overlap disabled"
        )
        self.seq_prefill_overlap = False
        return
    if self._capture_layer_ids[-1] >= self.model.end_layer - 1:
        logger.warning(
            "[SEQ_GUARD] J requested but final capture layer has no model tail; "
            "overlap disabled"
        )
        self.seq_prefill_overlap = False
        return
    owner_ref = weakref.ref(self)
    for layer_id in self._capture_layer_ids:
        layer = self.model.layers[layer_id]

        def hook(module, _args, output, captured_layer_id=layer_id):
            owner = owner_ref()
            if owner is not None:
                _prefill_overlap_hook(owner, captured_layer_id, module, output)

        self._prefill_overlap_hook_handles.append(
            layer.register_forward_hook(hook)
        )
    logger.info(
        f"[SEQ_GUARD] J sampled-prefill overlap hooks installed at "
        f"layers={self._capture_layer_ids}; tail starts after {self._capture_layer_ids[-1]}"
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
        if self.prefill_risk_mode == "all" and self.prefill_score_prob < 1.0:
            prepared = _prepare_sampled_prefill(
                self, feat_t, slot_ids, forward_batch
            )
            return _finish_sampled_prefill(self, prepared, forward_batch)

        hist_seq, hist_mask, ema = self.hist_buf.prefill_update(
            slot_ids=slot_ids,
            feat_seq=feat_t,
            seq_lens=forward_batch.extend_seq_lens,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            per_token=(self.prefill_risk_mode == "all"),
            sample_idx=None,
        )
        if self.prefill_risk_mode == "all":
            seq_lens = forward_batch.extend_seq_lens.to(torch.long)
            B = seq_lens.size(0)
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
        self._capture_layer_ids = []
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
        self.fused_sampled_ema = PGUARD_SEQ_FUSED_SAMPLED_EMA
        self.seq_fuse_kernels = PGUARD_SEQ_FUSE_KERNELS
        self.seq_compile = PGUARD_SEQ_COMPILE
        self.seq_decode_tp_fusion = PGUARD_SEQ_DECODE_TP_FUSION
        self._decode_tp_ctx = None
        self._guard_compiled = None

        self.row_shard = PGUARD_SEQ_ROW_SHARD
        if self.row_shard:
            from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

            if is_dsa_enable_prefill_cp():
                self.row_shard = False
                logger.warning(
                    "[SEQ_GUARD] PGUARD_SEQ_ROW_SHARD=1 但 CP prefill 已开启，"
                    "行切分不生效（CP 下各 rank 的 aux 只覆盖自己那片 token，"
                    "跨 rank 合并前提不成立）"
                )
        self._shard_ctx = None
        self._attn_world = None
        self._prefill_counter = 0
        self._score_gen = None

        self._seq_checkpoint_config = require_seq_guard_config(
            PGUARD_SEQ_STATE_PATH, "DeepSeek sequence checkpoint"
        )
        self.seq_guard_profile = resolve_seq_guard_profile(
            PGUARD_SEQ_STATE_PATH,
            explicit_version=PGUARD_VERSION,
        )
        static_layers = tuple(capture_layer_indices())
        n_cap = (
            len(static_layers)
            if static_layers
            else (self.model.end_layer - self.model.start_layer)
        )
        self._seq_d_feat = n_cap * config.hidden_size
        classification_heads = None
        if self._seq_checkpoint_config:
            if self._seq_checkpoint_config.get("pguard_version"):
                classification_heads = validate_production_classification_schema(
                    self._seq_checkpoint_config,
                    context="DeepSeek sequence checkpoint",
                )
                validate_seq_model_identity(
                    self._seq_checkpoint_config,
                    ("deepseek_v4",),
                    context="DeepSeek sequence checkpoint",
                    base_model_path=base_model_path,
                )
                validate_seq_feature_selection(
                    self._seq_checkpoint_config,
                    tuple(layer + 1 for layer in static_layers),
                    context="DeepSeek sequence checkpoint",
                )
            else:
                # Pre-metadata random/mock heads retain their historical schema.
                classification_heads = resolve_classification_heads(
                    self._seq_checkpoint_config
                )
                if classification_heads:
                    validate_task_schema(
                        self._seq_checkpoint_config,
                        classification_heads,
                        context="DeepSeek legacy sequence checkpoint",
                    )
            checkpoint_layers = resolve_checkpoint_capture_layers(
                self._seq_checkpoint_config, selected_layer_offset=-1
            )
            if checkpoint_layers and checkpoint_layers != static_layers:
                raise ValueError(
                    f"DeepSeek sequence checkpoint capture layers {checkpoint_layers} != "
                    f"PGUARD_CAPTURE_LAYERS {static_layers}"
                )
        self._seq_semantics = {
            "pguard_version": self.seq_guard_profile.version,
            "d_feat": self._seq_d_feat,
            "hidden_size": config.hidden_size,
            "n_cap": n_cap,
            "window_n": self.hist_n,
            "ema_update_mask": "attention_mask",
            "ema_scales": self.ema_scales,
            "n_bert_layers": self.seq_guard_profile.n_fusion_layers,
            "n_adapter_layers": self.seq_guard_profile.n_adapter_layers,
            "n_attn_heads": 8,
            "top_L": 3,
            "gate_first": self.seq_guard_profile.gate_first,
            "num_classes": (
                sum(size for _, size in classification_heads)
                if classification_heads
                else PGUARD_SEQ_NUM_CLASSES
            ),
        }
        validate_seq_checkpoint_semantics(
            self._seq_checkpoint_config,
            None,
            self._seq_semantics,
            context="DeepSeek sequence checkpoint",
            require_formal_config_metadata=True,
        )
        self.sequence_guard = SequencePluginGuard(
            d_feat=self._seq_d_feat,
            window_n=self.hist_n,
            n_cap=n_cap,
            n_scales=len(self.ema_scales),
            n_bert_layers=self.seq_guard_profile.n_fusion_layers,
            n_adapter_layers=self.seq_guard_profile.n_adapter_layers,
            num_classes=PGUARD_SEQ_NUM_CLASSES,
            dtype=torch.bfloat16,
            gate_topk=self.gate_topk,
            fuse_kernels=self.seq_fuse_kernels,
            fuse_classifiers=PGUARD_SEQ_FUSE_CLASSIFIERS,
            classification_heads=classification_heads,
        )
        logger.info(
            "[SEQ_GUARD] profile=%s fusion_layers=%d adapter_layers=%d gate_first=%s",
            self.seq_guard_profile.version,
            self.seq_guard_profile.n_fusion_layers,
            self.seq_guard_profile.n_adapter_layers,
            self.seq_guard_profile.gate_first,
        )
        self.hist_buf = None
        self.seq_prefill_overlap = PGUARD_SEQ_PREFILL_OVERLAP
        self._prefill_overlap_stream = None
        self._prefill_overlap_forward_batch = None
        self._prefill_overlap_features = []
        self._prefill_overlap_prepared = None
        self._prefill_overlap_done = None
        _install_prefill_overlap_hooks(self)

    def _score_generator(self) -> torch.Generator:
        if self._score_gen is None:
            self._score_gen = torch.Generator()  # CPU
        return self._score_gen

    def _setup_capture_layers(self) -> bool:
        """Configure DSpark post-layer captures using global physical layer IDs."""
        static_layer_ids = sorted(capture_layer_indices())
        if not self.pp_group.is_last_rank:
            self._capture_layer_ids = []
            return not static_layer_ids
        if not static_layer_ids:
            capture_layer_ids = list(range(self.model.start_layer, self.model.end_layer))
            self._capture_layer_ids = capture_layer_ids
            self.set_dspark_layers_to_capture(capture_layer_ids)
            return True
        missing = [
            layer
            for layer in static_layer_ids
            if not self.model.start_layer <= layer < self.model.end_layer
        ]
        if missing:
            raise ValueError(
                "all PGuard capture layers must live on the last PP rank; "
                f"missing={missing}, last-rank range="
                f"[{self.model.start_layer}, {self.model.end_layer})"
            )
        self._capture_layer_ids = static_layer_ids
        self.set_dspark_layers_to_capture(static_layer_ids)
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
                    context="DeepSeek sequence checkpoint",
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

        overlap_capture_layers = _begin_prefill_overlap(self, forward_batch)
        try:
            with _dsv4.get_attn_tp_context().maybe_input_scattered(forward_batch):
                hidden_states = self.model.forward(
                    input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
                )
        except Exception:
            if overlap_capture_layers is not None:
                _discard_prefill_overlap(self)
            raise
        finally:
            _restore_prefill_capture(self, overlap_capture_layers)
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if overlap_capture_layers is not None:
            aux_hidden_states = self._prefill_overlap_features
        hidden_states, pre_hc_head = hidden_states

        try:
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
        except Exception:
            if overlap_capture_layers is not None:
                _discard_prefill_overlap(self)
            raise
        if overlap_capture_layers is not None:
            guard_logits = _consume_prefill_overlap(self, forward_batch)
        else:
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


class DeepseekV4ForCausalLM(DeepseekV4PGuardForCausalLM):
    pass


