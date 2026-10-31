import importlib
from types import MethodType
import torch


def _cfc_gemma_allreduce(
    norm, x, residual=None, post_residual_addition=None, use_attn_tp_group=True
):
    from sglang.srt.distributed import (
        get_attn_tp_group,
        get_moe_ep_group,
        get_moe_tp_group,
    )
    from sglang.srt.layers import flashinfer_comm_fusion as fusion
    from sglang.srt.runtime_context import get_parallel, get_server_args

    parallel = get_parallel()
    group = (
        get_attn_tp_group()
        if use_attn_tp_group
        else get_moe_ep_group()
        if parallel.moe_ep_size > 1
        else get_moe_tp_group()
    )
    if residual is None or group.world_size == 1:
        return norm(x, residual, post_residual_addition)
    if post_residual_addition is not None:
        residual = residual + post_residual_addition
    if (
        x.dtype == torch.bfloat16
        and fusion._flashinfer_comm is not None
        and (
            fusion.resolve_flashinfer_allreduce_fusion_backend(get_server_args())
            == "trtllm"
        )
        and x.is_contiguous()
        and residual.is_contiguous()
        and norm.weight.is_contiguous()
        and fusion.ensure_workspace_initialized(
            max_token_num=max(x.shape[0], 2048),
            hidden_dim=x.shape[-1],
            dtype=x.dtype,
            token_num=x.shape[0],
            use_attn_tp_group=use_attn_tp_group,
        )
    ):
        workspace = fusion._get_workspace_manager(use_attn_tp_group).workspace
        if workspace is not None:
            residual_out = torch.empty_like(residual)
            norm_out = torch.empty_like(x)
            fusion._flashinfer_comm.allreduce_fusion(
                input=x,
                workspace=workspace,
                pattern=fusion._flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                trigger_completion_at_end=False,
                residual_out=residual_out,
                norm_out=norm_out,
                residual_in=residual,
                rms_gamma=norm.weight,
                rms_eps=norm.variance_epsilon,
                fp32_acc=True,
                weight_bias=1.0,
            )
            return (norm_out, residual_out)
    reduced = group.all_reduce(x.float()).to(x.dtype)
    return norm(reduced, residual, None)


def _install_cfc_gemma_precision(backbone):
    for layer in backbone.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name)
            norm._forward_method = norm.forward_native
            norm.forward_with_allreduce_fusion = MethodType(_cfc_gemma_allreduce, norm)


_cfc_original_gdn_prefill_conv = None
_cfc_original_gdn_prefill_gating = None
_cfc_original_gdn_prefill_intra = None
_cfc_hf_gdn_prefill_intra = None
_cfc_hf_rcp_ln2 = None


def _cfc_native_gdn_prefill_conv(
    x,
    weight,
    bias=None,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    conv_states=None,
    activation="silu",
    pad_slot_id=-1,
    **kwargs,
):
    if _cfc_original_gdn_prefill_conv is None:
        raise RuntimeError("CFC native GDN prefill convolution is not installed")
    return _cfc_original_gdn_prefill_conv(
        x.contiguous(),
        weight,
        bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states,
        activation=activation,
        pad_slot_id=pad_slot_id,
        **kwargs,
    )


def _install_cfc_native_gdn_prefill_conv():
    global _cfc_original_gdn_prefill_conv
    backend = importlib.import_module("sglang.srt.layers.attention.linear.gdn_backend")
    current = backend.causal_conv1d_fn
    if current is _cfc_native_gdn_prefill_conv:
        return
    if (
        _cfc_original_gdn_prefill_conv is not None
        and current is not _cfc_original_gdn_prefill_conv
    ):
        raise RuntimeError("GDN prefill convolution was replaced by another adapter")
    _cfc_original_gdn_prefill_conv = current
    backend.causal_conv1d_fn = _cfc_native_gdn_prefill_conv


def _cfc_native_gdn_prefill_gating(A_log, a, b, dt_bias):
    if _cfc_original_gdn_prefill_gating is None:
        raise RuntimeError("CFC GDN prefill gating is not installed")
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    return (g.unsqueeze(0), b.sigmoid().unsqueeze(0))


def _install_cfc_native_gdn_prefill_gating():
    global _cfc_original_gdn_prefill_gating
    backend = importlib.import_module("sglang.srt.layers.attention.linear.gdn_backend")
    current = backend.fused_gdn_gating
    if current is _cfc_native_gdn_prefill_gating:
        return
    if (
        _cfc_original_gdn_prefill_gating is not None
        and current is not _cfc_original_gdn_prefill_gating
    ):
        raise RuntimeError("GDN prefill gating was replaced by another adapter")
    _cfc_original_gdn_prefill_gating = current
    backend.fused_gdn_gating = _cfc_native_gdn_prefill_gating


def _cfc_hf_gdn_prefill_intra_adapter(
    k, v, g=None, beta=None, cu_seqlens=None, chunk_size=64, chunk_indices=None
):
    if _cfc_hf_gdn_prefill_intra is None or _cfc_hf_rcp_ln2 is None:
        raise RuntimeError("CFC HF GDN prefill intra is not installed")
    return _cfc_hf_gdn_prefill_intra(
        k=k,
        v=v,
        g=g * _cfc_hf_rcp_ln2,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )


def _install_cfc_hf_gdn_prefill_intra():
    global _cfc_original_gdn_prefill_intra
    global _cfc_hf_gdn_prefill_intra
    global _cfc_hf_rcp_ln2
    chunk = importlib.import_module("sglang.kernels.ops.attention.fla.chunk")
    current = chunk.chunk_gated_delta_rule_fwd_intra
    if current is _cfc_hf_gdn_prefill_intra_adapter:
        return
    if (
        _cfc_original_gdn_prefill_intra is not None
        and current is not _cfc_original_gdn_prefill_intra
    ):
        raise RuntimeError("GDN prefill intra was replaced by another adapter")
    hf_chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
    constants = importlib.import_module("fla.ops.utils.constant")
    _cfc_original_gdn_prefill_intra = current
    _cfc_hf_gdn_prefill_intra = hf_chunk.chunk_gated_delta_rule_fwd_intra
    _cfc_hf_rcp_ln2 = constants.RCP_LN2
    chunk.chunk_gated_delta_rule_fwd_intra = _cfc_hf_gdn_prefill_intra_adapter


def install_cfc_hf_aligned_backbone(backbone):
    try:
        layers = tuple(backbone.layers)
        if not layers:
            raise ValueError("backbone has no layers")
        for layer in layers:
            for name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, name)
                if not callable(norm.forward_native):
                    raise TypeError(f"{name}.forward_native is not callable")
                if not callable(norm.forward_with_allreduce_fusion):
                    raise TypeError(
                        f"{name}.forward_with_allreduce_fusion is not callable"
                    )
                for attr in ("_forward_method", "weight", "variance_epsilon"):
                    getattr(norm, attr)
        backend = importlib.import_module(
            "sglang.srt.layers.attention.linear.gdn_backend"
        )
        chunk = importlib.import_module("sglang.kernels.ops.attention.fla.chunk")
        hf_chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk_fwd")
        constants = importlib.import_module("fla.ops.utils.constant")
        mounts = (
            (
                backend,
                "causal_conv1d_fn",
                _cfc_native_gdn_prefill_conv,
                _cfc_original_gdn_prefill_conv,
            ),
            (
                backend,
                "fused_gdn_gating",
                _cfc_native_gdn_prefill_gating,
                _cfc_original_gdn_prefill_gating,
            ),
            (
                chunk,
                "chunk_gated_delta_rule_fwd_intra",
                _cfc_hf_gdn_prefill_intra_adapter,
                _cfc_original_gdn_prefill_intra,
            ),
        )
        for module, name, adapter, original in mounts:
            current = getattr(module, name)
            if not callable(current):
                raise TypeError(f"{module.__name__}.{name} is not callable")
            if (
                original is not None
                and current is not adapter
                and (current is not original)
            ):
                raise RuntimeError(
                    f"{module.__name__}.{name} was replaced by another adapter"
                )
        if not callable(hf_chunk.chunk_gated_delta_rule_fwd_intra):
            raise TypeError("HF chunk_gated_delta_rule_fwd_intra is not callable")
        if not isinstance(constants.RCP_LN2, (int, float)):
            raise TypeError("HF RCP_LN2 is not numeric")
        distributed = importlib.import_module("sglang.srt.distributed")
        runtime = importlib.import_module("sglang.srt.runtime_context")
        for module, names in (
            (
                distributed,
                ("get_attn_tp_group", "get_moe_ep_group", "get_moe_tp_group"),
            ),
            (runtime, ("get_parallel", "get_server_args")),
        ):
            for name in names:
                if not callable(getattr(module, name)):
                    raise TypeError(f"{module.__name__}.{name} is not callable")
        fusion = importlib.import_module("sglang.srt.layers.flashinfer_comm_fusion")
        getattr(fusion, "_flashinfer_comm")
        for name in (
            "resolve_flashinfer_allreduce_fusion_backend",
            "ensure_workspace_initialized",
            "_get_workspace_manager",
        ):
            if not callable(getattr(fusion, name)):
                raise TypeError(f"flashinfer_comm_fusion.{name} is not callable")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "PGUARD_CFC_HF_ALIGNED_BACKBONE=1 requires supported Qwen/GDN/HF mount points"
        ) from exc
    _install_cfc_native_gdn_prefill_conv()
    _install_cfc_native_gdn_prefill_gating()
    _install_cfc_hf_gdn_prefill_intra()
    _install_cfc_gemma_precision(backbone)
