import functools
import logging

from ..config import PGUARD_SKIP_FIRST_TOKEN
from . import guard_variant

logger = logging.getLogger(__name__)
_installed = False


def installed() -> bool:
    return _installed


def patch_forward_guard_cadence():
    global _installed
    if _installed:
        return
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ModuleNotFoundError as exc:
        logger.warning(
            "[patches] guard cadence hook unavailable: %s; "
            "打分退化为按 replay/eager 计步，首 token 必打分的保证失效",
            exc,
        )
        return

    original = ModelRunner.forward
    if getattr(original, "_pguard_forward_cadence", False):
        _installed = True
        guard_variant.note_hook_installed()
        return

    @functools.wraps(original)
    def forward_with_guard_cadence(self, forward_batch, *args, **kwargs):
        guard_variant.begin_forward(forward_batch.forward_mode.is_decode())
        return original(self, forward_batch, *args, **kwargs)

    forward_with_guard_cadence._pguard_forward_cadence = True
    ModelRunner.forward = forward_with_guard_cadence
    _installed = True
    guard_variant.note_hook_installed()
    if PGUARD_SKIP_FIRST_TOKEN:
        logger.warning(
            "PGUARD_SKIP_FIRST_TOKEN=1 时模型在 prefill 不算 guard logits，"
            "首 token 必打分的保证失效"
        )
    logger.info("[patches] guard cadence patch enabled")
