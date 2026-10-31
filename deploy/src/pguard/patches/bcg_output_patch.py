import copy
import dataclasses
import logging
import os

import torch

from . import guard_variant

logger = logging.getLogger(__name__)

_patched = False


def _is_dataclass_instance(x) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _contains_tensor(x) -> bool:
    if torch.is_tensor(x):
        return True
    if isinstance(x, dict):
        return any(_contains_tensor(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_contains_tensor(v) for v in x)
    if _is_dataclass_instance(x):
        return any(
            _contains_tensor(getattr(x, f.name, None)) for f in dataclasses.fields(x)
        )
    return False


def patch_bcg_output_types():
    global _patched
    if _patched or not guard_variant.enabled():
        return
    try:
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend as _Backend,
        )
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    except ImportError:
        return

    if _upstream_supports(_Backend):
        logger.info(
            "[bcg_output] sglang breakable backend already handles dataclass/dict "
            "outputs; patch skipped"
        )
        _patched = True
        return

    def _is_upstream_known(x) -> bool:
        return (
            x is None
            or torch.is_tensor(x)
            or isinstance(x, (PPProxyTensors, tuple, list))
        )

    def _takes_new_branch(x) -> bool:
        return not _is_upstream_known(x) and (
            _is_dataclass_instance(x) or isinstance(x, dict)
        )

    def _tensor_fields(obj):
        return [
            f.name
            for f in dataclasses.fields(obj)
            if _contains_tensor(getattr(obj, f.name, None))
        ]

    orig_output_rows = _Backend._output_rows
    orig_alloc = _Backend._alloc_full_buffer
    orig_slice = _Backend._slice_output
    orig_copy = _Backend._copy_output_to_buffer

    def _output_rows(self, output, cap: int) -> int:
        if _takes_new_branch(output):
            if isinstance(output, dict):
                values = list(output.values())
            else:
                values = [getattr(output, n) for n in _tensor_fields(output)]
            return min(
                [cap] + [self._output_rows(v, cap) for v in values if v is not None]
            )
        return orig_output_rows(self, output, cap)

    def _alloc_full_buffer(self, output, size: int):
        if _takes_new_branch(output):
            if isinstance(output, dict):
                return {
                    k: self._alloc_full_buffer(v, size) for k, v in output.items()
                }
            buffer = copy.copy(output)
            for name in _tensor_fields(output):
                setattr(
                    buffer, name, self._alloc_full_buffer(getattr(output, name), size)
                )
            return buffer
        return orig_alloc(self, output, size)

    def _slice_output(self, output, num_tokens: int):
        if _takes_new_branch(output):
            if isinstance(output, dict):
                return {
                    k: self._slice_output(v, num_tokens) for k, v in output.items()
                }
            sliced = copy.copy(output)
            for name in _tensor_fields(output):
                setattr(
                    sliced, name, self._slice_output(getattr(output, name), num_tokens)
                )
            return sliced
        return orig_slice(self, output, num_tokens)

    def _copy_output_to_buffer(self, output, output_buffer, num_tokens: int) -> None:
        if _takes_new_branch(output) or _takes_new_branch(output_buffer):
            if type(output) is not type(output_buffer):
                raise ValueError(
                    "BCG output structure changed between capture sizes: "
                    f"{type(output)} vs {type(output_buffer)}"
                )
            if isinstance(output, dict):
                if output.keys() != output_buffer.keys():
                    raise ValueError(
                        "BCG output dict structure changed between capture sizes: "
                        f"{sorted(output.keys())} != {sorted(output_buffer.keys())}"
                    )
                for key, value in output.items():
                    self._copy_output_to_buffer(
                        value, output_buffer[key], num_tokens
                    )
                return
            names, buffer_names = _tensor_fields(output), _tensor_fields(output_buffer)
            if names != buffer_names:
                raise ValueError(
                    "BCG output dataclass tensor fields changed between capture "
                    f"sizes: {names} != {buffer_names}"
                )
            for name in names:
                self._copy_output_to_buffer(
                    getattr(output, name), getattr(output_buffer, name), num_tokens
                )
            return
        return orig_copy(self, output, output_buffer, num_tokens)

    _Backend._output_rows = _output_rows
    _Backend._alloc_full_buffer = _alloc_full_buffer
    _Backend._slice_output = _slice_output
    _Backend._copy_output_to_buffer = _copy_output_to_buffer

    _patched = True
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════════╗\n"
        f"║  [PATCH] BCG dataclass/dict output support ENABLED (pid={os.getpid()}, ppid={os.getppid()})  ║\n"
        "╚══════════════════════════════════════════════════════╝\n"
        "\n"
    )


@dataclasses.dataclass
class _Probe:

    tensor: torch.Tensor
    mapping: dict
    plain: list
    empty: None = None


def _upstream_supports(backend_cls) -> bool:
    probe_self = object.__new__(backend_cls)
    rows = 2
    sample = _Probe(
        tensor=torch.zeros(rows, 3),
        mapping={"guard_logits": torch.zeros(rows, 4)},
        plain=[0.0, 1.0],
    )
    try:
        buffer = backend_cls._alloc_full_buffer(probe_self, sample, rows)
        out_rows = backend_cls._output_rows(probe_self, sample, rows)
        backend_cls._copy_output_to_buffer(probe_self, sample, buffer, out_rows)
        sliced = backend_cls._slice_output(probe_self, buffer, out_rows)
    except Exception:
        return False
    return (
        isinstance(buffer, _Probe)
        and isinstance(sliced, _Probe)
        and torch.is_tensor(sliced.tensor)
        and isinstance(sliced.mapping, dict)
        and sliced.mapping.keys() == sample.mapping.keys()
    )
