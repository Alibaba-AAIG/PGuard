from typing import Any, Optional
import torch


class CfcPrefillTpContext:
    __slots__ = ("rank", "world", "group")

    def __init__(self, rank: int, world: int, group: Optional[Any]) -> None:
        self.rank = rank
        self.world = world
        self.group = group


def resolve_cfc_prefill_tp_context() -> CfcPrefillTpContext:
    from sglang.srt.distributed import (
        get_attn_tensor_model_parallel_rank,
        get_attn_tensor_model_parallel_world_size,
        get_attn_tp_group,
    )

    world = int(get_attn_tensor_model_parallel_world_size())
    if world <= 1:
        return CfcPrefillTpContext(rank=0, world=world, group=None)
    return CfcPrefillTpContext(
        rank=int(get_attn_tensor_model_parallel_rank()),
        world=world,
        group=get_attn_tp_group(),
    )


def _is_cuda_graph_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_current_stream_capturing()


def run_cfc_prefill_tp_dedup(
    guard,
    feat: torch.Tensor,
    seq_lens: torch.Tensor,
    context: CfcPrefillTpContext,
    *,
    max_seq_len: int,
    seq_lens_host=None,
):
    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "CFC TP-deduplicated prefill does not support CUDA graph capture"
        )
    prefill_kwargs = {"max_seq_len": max_seq_len}
    if seq_lens_host is not None:
        prefill_kwargs["seq_lens_host"] = seq_lens_host
    if context.world <= 1:
        return guard.prefill_parallel(feat, seq_lens, **prefill_kwargs)
    if context.rank == 0:
        h0, query_logits, prefill_state = guard.prefill_parallel(
            feat, seq_lens, **prefill_kwargs
        )
        h0 = h0.contiguous()
        query_logits = query_logits.contiguous()
    else:
        batch_size = seq_lens.shape[0]
        h0 = torch.empty(
            (batch_size, guard.mem_dim),
            device=feat.device,
            dtype=next(guard.parameters()).dtype,
        )
        query_logits = torch.empty(
            (batch_size, guard.num_classes), device=feat.device, dtype=torch.float32
        )
        prefill_state = None
    context.group.broadcast(h0, src=0)
    context.group.broadcast(query_logits, src=0)
    return (h0, query_logits, prefill_state)
