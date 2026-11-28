# vLLM general-plugin entry point.
#
# load_general_plugins() executes this function in every vLLM process (engine
# core included), which is exactly where the custom architectures must be
# registered and the model-runner patch installed.

import logging

logger = logging.getLogger("pguard_vllm")


def register():
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        from .backbones import BackboneRegistry
        for adapter in BackboneRegistry.all():
            class_ref = f"{adapter.module}:{adapter.arch_name}"
            ModelRegistry.register_model(adapter.arch_name, class_ref)
            logger.info("pguard_vllm: registered architecture %s (%s)",
                        adapter.arch_name, adapter.name)
    except Exception as e:
        logger.error("pguard_vllm: architecture registration failed: %s", e)

    try:
        from .patch_runner import _install_patch
        _install_patch()
    except Exception as e:
        logger.error("pguard_vllm: model-runner patch failed: %s", e)
