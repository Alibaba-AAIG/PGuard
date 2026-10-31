"""Preserve customized_info across prefill CUDA graph output trimming."""

import functools
import logging

logger = logging.getLogger(__name__)
_installed = False


def patch_prefill_trim_customized_info():
    global _installed
    if _installed:
        return
    try:
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )
    except ModuleNotFoundError as exc:
        runner_module = "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
        if not (exc.name == runner_module or runner_module.startswith(exc.name + ".")):
            raise
        logger.warning("[patches] prefill trim hook unavailable: %s", exc)
        return

    original = getattr(PrefillCudaGraphRunner, "_trim_logits_output", None)
    if original is None:
        return
    if getattr(original, "_pguard_prefill_trim", False):
        _installed = True
        return

    @functools.wraps(original)
    def trim_with_customized_info(self, output):
        trimmed = original(self, output)
        customized = getattr(output, "customized_info", None)
        if customized is not None and trimmed is not None:
            trimmed.customized_info = customized
        return trimmed

    trim_with_customized_info._pguard_prefill_trim = True
    PrefillCudaGraphRunner._trim_logits_output = trim_with_customized_info
    _installed = True
    logger.info("[patches] prefill trim customized_info patch enabled")
