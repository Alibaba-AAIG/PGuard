# vLLM general-plugin entry point.
#
# load_general_plugins() executes this function in every vLLM process (engine
# core included), which is exactly where the custom architecture must be
# registered and the model-runner patch installed.

import logging

logger = logging.getLogger("pguard_vllm")

_ARCH = "Qwen3_5PGuardForConditionalGeneration"


def register():
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        ModelRegistry.register_model(
            _ARCH,
            "pguard_vllm.model:Qwen3_5PGuardForConditionalGeneration",
        )
        logger.info("pguard_vllm: registered architecture %s", _ARCH)
    except Exception as e:
        logger.error("pguard_vllm: architecture registration failed: %s", e)

    try:
        from .patch_runner import _install_patch
        _install_patch()
    except Exception as e:
        logger.error("pguard_vllm: model-runner patch failed: %s", e)
