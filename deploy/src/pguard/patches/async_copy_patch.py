import functools
import logging
import os

import torch

from ..config import PGUARD_ASYNC_COPY as _ASYNC_COPY

logger = logging.getLogger(__name__)


class PendingGuardCopy:

    _pool = {}
    _pool_bytes = 0
    _retired = []
    _retired_bytes = 0
    _max_cached_bytes = 8 * 1024 * 1024
    _max_buffers_per_key = 2

    def __init__(self, tensor: torch.Tensor):
        if not tensor.is_cuda:
            raise ValueError("PendingGuardCopy requires a CUDA tensor")
        type(self)._reap_retired()
        self.source = tensor
        self.key = (tuple(tensor.shape), tensor.dtype)
        self.nbytes = tensor.numel() * tensor.element_size()
        bufs = self._pool.setdefault(self.key, [])
        if bufs:
            self.pinned = bufs.pop()
            type(self)._pool_bytes -= self.nbytes
        else:
            self.pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        self.stream = torch.cuda.current_stream(tensor.device)
        self.pinned.copy_(tensor, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record(self.stream)
        self._values = None
        self._retired_copy = False
        self._discarded = False

    @classmethod
    def _reap_retired(cls):
        remaining = []
        for pending in cls._retired:
            if pending.event.query():
                cls._retired_bytes -= pending.nbytes
                pending._complete_discard()
            else:
                remaining.append(pending)
        cls._retired = remaining
        while cls._retired_bytes > cls._max_cached_bytes and cls._retired:
            pending = cls._retired.pop(0)
            pending.event.synchronize()
            cls._retired_bytes -= pending.nbytes
            pending._complete_discard()

    def _release_buffer(self):
        pinned = self.pinned
        if pinned is None:
            return
        bufs = self._pool.setdefault(self.key, [])
        cls = type(self)
        if (
            len(bufs) < cls._max_buffers_per_key
            and cls._pool_bytes + self.nbytes <= cls._max_cached_bytes
        ):
            bufs.append(pinned)
            cls._pool_bytes += self.nbytes
        self.pinned = None

    def _complete_discard(self):
        self._release_buffer()
        self.source = None
        self.event = None
        self.stream = None
        self._discarded = True

    def resolve(self):
        if self._values is not None:
            return self._values
        if self._discarded or self._retired_copy:
            raise RuntimeError("discarded guard copy cannot be resolved")
        self.event.synchronize()
        self._values = self.pinned.tolist()
        self._release_buffer()
        self.source = None
        self.event = None
        self.stream = None
        return self._values

    def retire(self):
        if self._values is not None or self._discarded or self._retired_copy:
            return
        if self.event.query():
            self._complete_discard()
            return
        self._retired_copy = True
        cls = type(self)
        cls._retired.append(self)
        cls._retired_bytes += self.nbytes
        cls._reap_retired()


_installed = False
_cleanup_installed = False

_HOOK_CANDIDATES = (
    (
        "sglang.srt.managers.scheduler_components.batch_result_processor",
        "SchedulerBatchResultProcessor",
        "_maybe_collect_customized_info",
    ),
    (
        "sglang.srt.managers.scheduler_output_processor_mixin",
        "SchedulerOutputProcessorMixin",
        "maybe_collect_customized_info",
    ),
)
_RESULT_PROCESSORS = (
    "process_batch_result_prefill",
    "process_batch_result_decode",
)


def installed() -> bool:
    return _installed and _cleanup_installed


def installation_status():
    return {"collector": _installed, "cleanup": _cleanup_installed}


def _resolve_hook():
    import importlib

    for mod_name, cls_name, attr in _HOOK_CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None and hasattr(cls, attr):
            return cls, attr
    return None, None


def _iter_pending(customized_info):
    for value in (customized_info or {}).values():
        if isinstance(value, PendingGuardCopy):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, PendingGuardCopy):
                    yield item


def _resolve_pending(customized_info):
    for key, value in list((customized_info or {}).items()):
        if isinstance(value, PendingGuardCopy):
            customized_info[key] = value.resolve()
        elif isinstance(value, list):
            customized_info[key] = [
                item.resolve() if isinstance(item, PendingGuardCopy) else item
                for item in value
            ]


def _install_result_cleanup(cls):
    installed = set()
    for attr in _RESULT_PROCESSORS:
        original = getattr(cls, attr, None)
        if original is None:
            continue
        if getattr(original, "_pguard_async_cleanup", False):
            installed.add(attr)
            continue

        @functools.wraps(original)
        def process_with_cleanup(self, batch, result, *args, __original=original, **kwargs):
            logits_output = getattr(result, "logits_output", None)
            pending = list(
                _iter_pending(getattr(logits_output, "customized_info", None))
            )
            try:
                return __original(self, batch, result, *args, **kwargs)
            finally:
                for copy in pending:
                    copy.retire()

        process_with_cleanup._pguard_async_cleanup = True
        setattr(cls, attr, process_with_cleanup)
        installed.add(attr)
    return installed == set(_RESULT_PROCESSORS)


def patch_maybe_collect_async_copy():
    global _installed, _cleanup_installed
    if not _ASYNC_COPY or installed():
        return
    cls, attr = _resolve_hook()
    if cls is None:
        logger.error(
            "PGUARD_ASYNC_COPY=1 但找不到 maybe_collect_customized_info 挂载点"
            f"（试过 {[candidate[0] for candidate in _HOOK_CANDIDATES]}），"
            "sampler 将退回同步拷贝"
        )
        return
    original = getattr(cls, attr)
    if not getattr(original, "_pguard_async_resolve", False):

        @functools.wraps(original)
        def collect_with_resolve(self, i, req, logits_output):
            if logits_output is not None and logits_output.customized_info:
                _resolve_pending(logits_output.customized_info)
            return original(self, i, req, logits_output)

        collect_with_resolve._pguard_async_resolve = True
        setattr(cls, attr, collect_with_resolve)
    _cleanup_installed = _install_result_cleanup(cls)
    _installed = True
    if not _cleanup_installed:
        logger.error(
            "PGUARD_ASYNC_COPY=1 但找不到 batch result cleanup 挂载点，"
            "sampler 将退回同步拷贝"
        )
        return
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        f"║  [PATCH] guard async copy ENABLED (pid={os.getpid()}, ppid={os.getppid()})  ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
    )
