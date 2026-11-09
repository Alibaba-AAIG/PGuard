import logging
import os
from functools import partial
from sglang.srt.layers.sampler import Sampler, register_sampler_backend
import torch
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from pguard.config import (
    is_debug,
    PGUARD_OVERWRITE_MODEL,
    PGUARD_ASYNC_COPY as _ASYNC_COPY,
)
from pguard.patches import guard_variant
from pguard.patches import async_copy_patch
from pguard.patches.async_copy_patch import PendingGuardCopy
from pguard_server_plugin import POLICY_ENV

logger = logging.getLogger(__name__)


def customized_info_to_cpu(customized_info: dict, need_copy=True, batch_size=None):
    if customized_info is None:
        return None
    final_customized_info = {}
    if not need_copy:
        return final_customized_info

    def to_cpu(t: torch.Tensor):
        if not isinstance(t, torch.Tensor):
            return t
        return t.to("cpu").tolist()

    for key in list(customized_info.keys()):
        if isinstance(customized_info[key], torch.Tensor):
            if is_debug():
                logger.info(f"key: {key}, shape: {customized_info[key].shape}")
            if need_copy:
                final_customized_info[key] = to_cpu(customized_info[key])
        elif isinstance(customized_info[key], list):
            values = customized_info[key]
            if need_copy:
                copied_values = []
                for i, v in enumerate(values):
                    copied_values.append(to_cpu(v))
                final_customized_info[key] = copied_values
        if (
            need_copy
            and isinstance(customized_info[key], (torch.Tensor, list))
            and (batch_size is not None)
        ):
            if is_debug():
                logger.warning(
                    f"batch_size: {batch_size}, {key}: len_{len(final_customized_info[key])}"
                )
            if len(customized_info[key]) > batch_size:
                final_customized_info[key] = final_customized_info[key][:batch_size]
            elif len(customized_info[key]) < batch_size:
                logger.warning(
                    f"pop key: {key}, len: {len(final_customized_info[key])}) != batch_size={batch_size!r}"
                )
                final_customized_info.pop(key)
    return final_customized_info


def customized_info_to_async(customized_info: dict, batch_size=None, pending_copy=PendingGuardCopy):
    if customized_info is None:
        return None
    final_customized_info = {}

    def prepare_tensor(tensor, key):
        if not tensor.is_cuda:
            return tensor.to("cpu").tolist()
        factory = pending_copy if key == "guard_logits" else PendingGuardCopy
        return factory(tensor)

    for key, value in customized_info.items():
        if not isinstance(value, (torch.Tensor, list)):
            continue
        if batch_size is not None:
            if len(value) > batch_size:
                value = value[:batch_size]
            elif len(value) < batch_size:
                logger.warning(
                    f"pop key: {key}, len: {len(value)} != batch_size={batch_size!r}"
                )
                continue
        if isinstance(value, torch.Tensor):
            final_customized_info[key] = prepare_tensor(value, key)
        else:
            final_customized_info[key] = [
                prepare_tensor(item, key) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
    return final_customized_info


class CustomSampler(Sampler):
    def __init__(self):
        super().__init__()
        self.is_guard_overwrite_model = PGUARD_OVERWRITE_MODEL
        self.async_copy = _ASYNC_COPY and async_copy_patch.installed()
        self.copy_to_cpu = customized_info_to_cpu
        self.copy_to_async = customized_info_to_async
        if os.getenv(POLICY_ENV):
            from pguard.interrupt_runtime import (
                copy_with_guard_cpu,
                install_interrupt,
                interrupt_pending_copy_class,
            )

            install_interrupt()
            self.copy_to_cpu = partial(copy_with_guard_cpu, customized_info_to_cpu)
            self.copy_to_async = partial(
                customized_info_to_async, pending_copy=interrupt_pending_copy_class()
            )

    def forward(
        self,
        logits_output: "LogitsProcessorOutput",
        sampling_info: "SamplingBatchInfo",
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
        **kwargs,
    ):
        if is_debug():
            logger.info(
                f"logits.shape: logits_output.next_token_logits.shape={logits_output.next_token_logits.shape!r}"
            )
        batch_size = logits_output.next_token_logits.shape[0]
        need_copy = False
        if (
            hasattr(logits_output, "customized_info")
            and logits_output.customized_info is not None
        ):
            need_copy = guard_variant.consume_decision()
            customized_info = logits_output.customized_info
            if self.async_copy and need_copy:
                logits_output.customized_info = self.copy_to_async(
                    customized_info, batch_size=batch_size
                )
            else:
                logits_output.customized_info = self.copy_to_cpu(
                    customized_info, need_copy=need_copy, batch_size=batch_size
                )
            contain_guard_logits = "guard_logits" not in logits_output.customized_info
            if (
                self.is_guard_overwrite_model
                and "guard_logits" not in logits_output.customized_info
            ):
                logits_output.customized_info["guard_logits"] = [0.0] * batch_size
            elif contain_guard_logits:
                if is_debug():
                    logger.info(
                        f"contain guard logits, logits type: {type(logits_output.customized_info['guard_logits'])}"
                    )
        elif self.is_guard_overwrite_model:
            if is_debug():
                logger.error(f"miss guard_logits info")
            logits_output.customized_info = {"guard_logits": [0.0] * batch_size}
        return super().forward(
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
            **kwargs,
        )


def create_custom_sampler():
    return CustomSampler()


from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES

for backend in SAMPLING_BACKEND_CHOICES:
    register_sampler_backend(backend, create_custom_sampler)
register_sampler_backend("guard_sampler", create_custom_sampler)
