"""Applies the interrupt policy once engine results are on the CPU; native finish logic cleans up."""

from contextvars import ContextVar
import functools
import importlib
import json
import math
import os

from pguard_server_plugin import POLICY_ENV
from .interrupt_policy import (
    STAGES,
    GuardScoreRows,
    InterruptHit,
    NonFiniteGuardScores,
    is_unscored,
    load_policy,
)


_active_result = ContextVar("pguard_interrupt_result", default=None)


def modeling_variant():
    return os.getenv("PGUARD_MODELING_VARIANT", "").strip().lower()


def validate_runtime(server_args, policy):
    if not policy.enabled:
        return
    if (os.getenv("SGLANG_EXTERNAL_MODEL_PACKAGE") != "pguard.sglang"
            or os.getenv("PGUARD_OVERWRITE_MODEL") != "1"):
        raise ValueError("interrupt_config requires pguard.sglang and PGUARD_OVERWRITE_MODEL=1")
    probability = float(os.getenv("PGUARD_COPY_LOGITS_PROB", "0.1"))
    if not math.isfinite(probability) or probability <= 0:
        raise ValueError("interrupt_config requires guard scoring; PGUARD_COPY_LOGITS_PROB must be > 0")
    if policy.groups["prefill"] and bool(int(os.getenv("PGUARD_SKIP_FIRST_TOKEN", "0"))):
        raise ValueError("prefill rules require PGUARD_SKIP_FIRST_TOKEN=0")
    if getattr(server_args, "is_embedding", False) or getattr(server_args, "encoder_only", False):
        raise ValueError("interrupt_config supports generation requests only")
    if getattr(server_args, "speculative_algorithm", None):
        raise ValueError("interrupt_config does not support speculative decoding")
    if getattr(server_args, "pp_size", 1) != 1:
        raise ValueError("interrupt_config does not support pipeline parallel")
    if getattr(server_args, "disaggregation_mode", "null") != "null":
        raise ValueError("interrupt_config does not support prefill/decode disaggregation")


@functools.lru_cache(maxsize=1)
def _policy_from_payload(payload):
    data = json.loads(payload)
    return load_policy(data["rules"], data["head_config"])


def configured_policy():
    payload = os.getenv(POLICY_ENV)
    if not payload:
        return None
    return _policy_from_payload(payload)


def _cpu_array(tensor):
    import torch

    # NumPy has no bfloat16; only the CPU copy is upcast, the GPU tensor keeps its dtype.
    return tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()


def copy_with_guard_cpu(original, info, need_copy=True, batch_size=None):
    import torch

    value = (info or {}).get("guard_logits")
    if need_copy and isinstance(value, torch.Tensor):
        cpu = value.to("cpu")
        copied_info = dict(info)
        copied_info["guard_logits"] = cpu
        result = original(copied_info, need_copy=need_copy, batch_size=batch_size)
        rows = result.get("guard_logits")
        if isinstance(rows, list):
            result["guard_logits"] = GuardScoreRows(rows, _cpu_array(cpu)[:len(rows)])
        return result
    return original(info, need_copy=need_copy, batch_size=batch_size)


def interrupt_pending_copy_class():
    from .patches.async_copy_patch import PendingGuardCopy

    class InterruptPendingCopy(PendingGuardCopy):
        # Separate pool and counters so these buffers never mix with other customized_info copies.
        _pool = {}
        _pool_bytes = 0
        _retired = []
        _retired_bytes = 0

        def _materialize(self):
            # The pinned buffer returns to the pool next, so the array must be a private copy.
            return GuardScoreRows(self.pinned.tolist(), _cpu_array(self.pinned).copy())

    return InterruptPendingCopy


class _BatchDecision:
    def __init__(self, policy, batch, result, stage, finish_reason):
        self.policy = policy
        self.batch = batch
        self.result = result
        self.stage = stage
        self.finish_reason = finish_reason
        self.applied = False
        # The native loop decrements chunk counters, so snapshot middle chunks before it runs.
        self.middle_req_ids = {
            id(req) for req in batch.reqs
            if getattr(req, "inflight_middle_chunks", getattr(req, "is_chunked", 0)) > 0
        } if stage == "prefill" else set()

    def apply(self):
        if self.applied:
            return
        self.applied = True
        logits = getattr(self.result, "logits_output", None)
        info = getattr(logits, "customized_info", None)
        if not info or "guard_logits" not in info:
            return
        rows = info["guard_logits"]
        if isinstance(rows, list) and is_unscored(rows):
            return
        decoded = {id(req) for req in (getattr(self.batch, "decoding_reqs", None) or [])}
        selected = {stage: [] for stage in STAGES}
        for index, req in enumerate(self.batch.reqs):
            if req.finished() or req.is_retracted or req.to_finish is not None:
                continue
            if id(req) in self.middle_req_ids:
                continue
            stage = "decode" if self.stage == "decode" or id(req) in decoded else "prefill"
            if self.policy.groups[stage]:
                selected[stage].append(index)
        if not any(selected.values()):
            return
        if hasattr(rows, "resolve"):
            rows = rows.resolve()
            info["guard_logits"] = rows
        if len(rows) != len(self.batch.reqs):
            raise RuntimeError("guard score rows do not match the batch requests")
        for stage, indices in selected.items():
            if not indices:
                continue
            subset = rows
            if len(indices) != len(rows):
                subset = [rows[index] for index in indices]
                if isinstance(rows, GuardScoreRows):
                    subset = GuardScoreRows(subset, rows.array[indices])
            try:
                hits = self.policy.match(subset, stage)
            except NonFiniteGuardScores:
                # A NaN in one request must abort only that request, not the whole batch.
                hits = []
                for row in subset:
                    try:
                        hits.append(self.policy.match([row], stage)[0])
                    except NonFiniteGuardScores as exc:
                        hits.append(InterruptHit(f"{stage}.invalid_guard_scores: {exc}", 0.0, 0.0))
            for index, hit in zip(indices, hits):
                if hit is not None:
                    self.batch.reqs[index].to_finish = self.finish_reason(message=hit.message)


def install_interrupt(policy=None):
    if policy is None:
        policy = configured_policy()
        if policy is not None:
            from sglang.srt.server_args import get_global_server_args

            # Re-check in the scheduler so a policy that bypassed the CLI cannot bypass validation.
            validate_runtime(get_global_server_args(), policy)
    if policy is None or not policy.enabled:
        return
    candidates = (
        ("sglang.srt.managers.scheduler_components.batch_result_processor", "SchedulerBatchResultProcessor"),
        ("sglang.srt.managers.scheduler_output_processor_mixin", "SchedulerOutputProcessorMixin"),
    )
    processor = None
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(exc.name + "."):
                continue
            raise
        processor = getattr(module, class_name, None)
        if processor is not None:
            break
    if processor is None:
        raise RuntimeError("this SGLang version has no supported batch result processor")
    if getattr(processor, "_pguard_interrupt_installed", False):
        return
    from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req

    finish_name = "update_finish_state" if hasattr(Req, "update_finish_state") else "check_finished"
    original_finish = getattr(Req, finish_name)

    @functools.wraps(original_finish)
    def finish_with_policy(self, *args, **kwargs):
        decision = _active_result.get()
        if decision is not None:
            decision.apply()
        return original_finish(self, *args, **kwargs)

    def result_wrapper(original, stage):
        @functools.wraps(original)
        def process(self, batch, result, *args, **kwargs):
            decision = _BatchDecision(policy, batch, result, stage, FINISH_ABORT)
            token = _active_result.set(decision)
            try:
                # The native method finishes copy_done and next_token_ids before calling finish.
                return original(self, batch, result, *args, **kwargs)
            finally:
                _active_result.reset(token)
        return process

    setattr(Req, finish_name, finish_with_policy)
    for stage in STAGES:
        name = f"process_batch_result_{stage}"
        setattr(processor, name, result_wrapper(getattr(processor, name), stage))
    processor._pguard_interrupt_installed = True
