"""SGLang hidden-state extractor with a transformer-like interface.

Hidden states must be complete and aligned with each request. Multi-round
chunked prefill breaks this contract, so the batch must finish in one prefill:
- context_length limits each request and KV capacity, not the batch token total.
- chunked_prefill_size limits tokens per forward; None keeps the SGLang default
  and -1 disables chunking. max_prefill_tokens still caps the forward.
- Radix cache is disabled so each request computes its full sequence.

Usage::

    with SGLangModel("Qwen/Qwen3-0.6B", max_model_len=8192) as model:
        out = model(input_ids=[[1, 2, 3]], output_hidden_states=True)
        # out.hidden_states: one [batch, seq_len, hidden_size] tensor per layer

For N transformer layers, captures comprise embedding output (0), residual
streams (1..N-1), and final norm output (N). layers selects a subset.
TP and independent DP ranks are supported; PP is not. Supported architectures
are listed in patch_sglang_all_layers.SUPPORTED_MODEL_TYPES.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

# AMD (ROCm) compatibility: env vars must be set before importing sglang
if torch.version.hip is not None:
    os.environ.setdefault('USE_ROCM_AITER_ROPE_BACKEND', '0')
    os.environ.setdefault('SGLANG_USE_AITER', '1')

@dataclass
class SGLangModelOutput:
    """Captured layers as [batch, seq_len, hidden_size] tensors.

    Batched sequences must have equal padded lengths for stacking.
    """

    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


def patch_preprocessor_config_file(model_path, max_pixels: int, min_pixels: int):
    """Align local preprocessor_config.json with input image-area constraints.

    SGLang rebuilds image tokens from this file; inconsistent values misalign
    hidden states with input_ids. GLM also ignores max_pixels/min_pixels kwargs.
    Updates size.longest_edge/shortest_edge and creates a first-run backup.
    Callers separately update any AutoProcessor instance already loaded.
    """
    import json
    import shutil
    import sys

    preprocessor_path = os.path.join(
        model_path, "preprocessor_config.json"
    )

    if not os.path.exists(preprocessor_path):
        print(
            f"[SGLangModel] WARNING: no local preprocessor_config.json found; "
            f"image preprocessing was not aligned.\n  path: {preprocessor_path}",
            file=sys.stderr,
            flush=True,
        )
        return

    with open(preprocessor_path, 'r') as f:
        cfg = json.load(f)

    size = cfg.get("size", {})
    longest = size.get("longest_edge", None)
    shortest = size.get("shortest_edge", None)

    # top-level max_pixels/min_pixels fallback (backwards-compatible keys)
    if longest is None:
        longest = cfg.get("max_pixels", None)
    if shortest is None:
        shortest = cfg.get("min_pixels", None)

    need_patch = (longest != max_pixels) or (shortest != min_pixels)

    if need_patch:
        # back up the original config (first run only, so reruns never overwrite the backup)
        backup_path = preprocessor_path + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(preprocessor_path, backup_path)

        # write the modified config
        cfg["size"] = {
            "longest_edge": max_pixels,
            "shortest_edge": min_pixels,
        }
        cfg.pop("max_pixels", None)
        cfg.pop("min_pixels", None)

        with open(preprocessor_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        print(
            f"[SGLangModel] WARN: VL model preprocessor_config.json auto-aligned.\n"
            f"  model path: {model_path}\n"
            f"  original: longest_edge={longest}, shortest_edge={shortest}\n"
            f"  patched: longest_edge={max_pixels} , "
            f"shortest_edge={min_pixels} \n"
            f"  backup: {backup_path}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[SGLangModel] OK: preprocessor_config.json already aligned "
            f"(longest_edge={longest}, shortest_edge={shortest})",
            file=sys.stderr,
            flush=True,
        )


class SGLangModel:
    """Patch SGLang, create an engine, and extract hidden states.

    shutdown() closes the engine but leaves source patches installed; run
    patch_sglang_all_layers.py --uninstall to restore them. VL extraction
    requires a local writable model directory because preprocessing alignment
    may update preprocessor_config.json and create a backup.

    layers accepts None/"all", comma-separated indices, or list[int].
    chunked_prefill_size=None keeps the engine default; -1 disables chunking.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        gpu_memory_utilization: float = None,
        tp_size: int = 1,
        layers: Union[None, str, List[int]] = None,
        chunked_prefill_size: int = None,
        attention_backend: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        padding_side: str = "left",
        engine_extra_args: dict = None,
    ):
        # ── 0. Allow nested event loops (Jupyter / IPython only) ─────
        try:
            # nest_asyncio is only for an existing interactive event loop;
            # applying it to SGLang's uvloop would fail.
            import asyncio

            asyncio.get_running_loop()  # raises RuntimeError if none
            import nest_asyncio

            nest_asyncio.apply()
        except (ImportError, RuntimeError):
            pass

        # ── 1. Load model config ─────────────────────────────────────
        from src.sgl.patch_sglang_all_layers import load_hf_config
        self.config = load_hf_config(model_path, trust_remote_code=True)
        self._forward_count = 0
        self._mismatch_count = 0
        self._is_vl = hasattr(self.config, "text_config")

        # VL models: make sure preprocessor_config.json matches the training pipeline
        if self._is_vl:
            self._patch_preprocessor_config(model_path, max_pixels, min_pixels)

        self.model_path = model_path
        _cfg = self.config.text_config if self._is_vl else self.config
        self.num_layers = _cfg.num_hidden_layers
        self.hidden_size = _cfg.hidden_size

        # padding handling: read pad_token from tokenizer_config.json (via
        # AutoTokenizer), falling back to eos_token_id. forward() strips pads
        # internally per self.padding_side -> runs the engine -> pads zeros back
        # per direction, so pads never pollute valid tokens (SGLang takes no
        # attention_mask) and RoPE positions stay correct.
        if padding_side not in ("left", "right"):
            raise ValueError(
                f"[SGLangModel] padding_side must be 'left' or 'right', got {padding_side!r}"
            )
        self.padding_side = padding_side
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _pad_id = _tokenizer.pad_token_id
        if _pad_id is None:
            _pad_id = _tokenizer.eos_token_id
        if _pad_id is None:
            raise RuntimeError(
                "[SGLangModel] cannot resolve pad_token_id from tokenizer "
                "(tokenizer_config.json missing pad_token / eos_token)."
            )
        self.pad_token_id = int(_pad_id)

        # validate that selected_layers stays within the model layer range [0, num_layers]
        if layers is not None and layers != "all":
            if isinstance(layers, str):
                _layer_list = [int(x.strip()) for x in layers.split(",") if x.strip().isdigit()]
            else:
                _layer_list = list(layers)
            _invalid = [x for x in _layer_list if x < 0 or x > self.num_layers]
            if _invalid:
                raise ValueError(
                    f"[SGLangModel] selected_layers out of range!"
                    f" model num_hidden_layers={self.num_layers}, valid range [0, {self.num_layers}],"
                    f" got invalid layer indices: {_invalid}"
                )

        # Determine captured layer indices
        _cap = os.environ.get("SGLANG_CAPTURE_LAYERS", "all")
        if _cap.strip().lower() == "all":
            self._captured_indices = list(range(self.num_layers + 1))
        else:
            self._captured_indices = sorted(
                int(x)
                for x in _cap.split(",")
                if x.strip().isdigit() and 0 <= int(x.strip()) <= self.num_layers
            )

        # ── 2. ROCm compatibility ──
        if torch.version.hip is not None:
            if attention_backend == 'triton':
                # triton still calls some aiter components, which cost extra GPU memory
                pass
            elif attention_backend == 'triton_noaiter':
                # triton with aiter fully disabled; requires the vllm layernorm patch (tested on pytorch280+sglang055 only)
                os.environ['SGLANG_USE_AITER'] = '0'
                os.environ['_SGLANG_VLLM_ROCM_COMPAT'] = '1'
                attention_backend = 'triton'
            elif attention_backend == 'aiter':
                pass
            elif attention_backend == 'aiter_lowmem':
                # aiter + per-expert shuffle to cut the loading-stage memory peak (2x -> ~1x; tested on pytorch280+sglang055 only)
                os.environ['_SGLANG_AITER_LOWMEM'] = '1'
                attention_backend = 'aiter'

        # ── 2.5 DeepSeek-V4 (sglang 0.5.13) kernel-path envs ──
        # gfx942 has no native FP4 instructions -> FP8 weights; deep_gemm is CUDA-only -> aiter/tilelang substitutes
        if getattr(self.config, "model_type", None) == "deepseek_v4":
            os.environ.setdefault("SGLANG_DSV4_FP4_EXPERTS", "0")      # FP8 experts (no FP4 hardware on gfx942)
            os.environ.setdefault("SGLANG_OPT_FP8_WO_A_GEMM", "0")     # no deep_gemm (CUDA-only)
            os.environ.setdefault("SGLANG_OPT_USE_AITER_INDEXER", "1")  # aiter indexer
            os.environ.setdefault("SGLANG_TOPK_TRANSFORM_512_TORCH", "1")  # torch topk, avoids CUDA jit
            os.environ.setdefault("SGLANG_OPT_USE_TOPK_V2", "0")       # default True invokes the CUDA jit plan_topk_v2
            os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")  # no ROCm port of deep_gemm
            os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "1")   # tilelang mhc main path
            os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "1")
            # the pytorch280 image ships triton 3.4 < the 3.6 required by aiter gluon (the gluon kernel is unused on this path; downgrade warning only)
            os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

        # ── 3. Validate model + patch sglang source ───────────────
        import src.sgl.patch_sglang_all_layers as patch_sglang_all_layers

        patch_sglang_all_layers.check_model_support(model_path)
        patch_sglang_all_layers.auto_patch(layers=layers)

        # ── 4. Create engine ─────────────────────────────────────────
        import sglang as sgl

        # Torchelastic variables can make the scheduler misdetect its TCPStore
        # role and hang during local launch. Remove them while creating the
        # engine, then restore them for the parent's subsequent DDP behavior.
        _TORCHELASTIC_ENV_KEYS = (
            "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_ERROR_FILE", "TORCH_ELASTIC_SUBPROCESS_HEARTBEAT_SECS",
            "GROUP_RANK", "GROUP_WORLD_SIZE",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
            "LOCAL_WORLD_SIZE",
        )
        _saved_env = {k: os.environ.pop(k) for k in _TORCHELASTIC_ENV_KEYS if k in os.environ}

        # ── 2.6 Extra engine args (dsv4 defaults cut the host-memory peak) ──
        _engine_extra = dict(engine_extra_args or {})
        if getattr(self.config, "model_type", None) == "deepseek_v4":
            # 294GB FP8 weights x 8 loading processes on one host would blow the
            # container's host-memory limit (OOM -9):
            # drop_cache_after_load drops the page cache right after loading;
            # prefetch threads 4->1 shrinks the IO buffer
            _engine_extra.setdefault("weight_loader_drop_cache_after_load", True)
            _engine_extra.setdefault("weight_loader_prefetch_num_threads", 1)
        
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            enable_return_hidden_states=True,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            context_length=max_model_len + 8,  # context_length must exceed the max per-sample length in a batch
            chunked_prefill_size=chunked_prefill_size,
            mem_fraction_static=gpu_memory_utilization,
            attention_backend=attention_backend,
            disable_cuda_graph=True,
            **_engine_extra,
        )

        os.environ.update(_saved_env)  # restore, so the parent process's later DDP behavior is unaffected

        # ── 5. Warmup ────────────────────────────────────────────────
        if self._is_vl:
            # mixed image-text batch warmup: one image sample + one text-only
            # sample, warming up the vision encoder, the LEGACY-PATH image-block
            # folding, the text-only branch, and SGLang's modality-splitting
            # dispatcher for mixed batches, all at once.
            from PIL import Image as _PILImage
            from transformers import AutoProcessor
            # a synthetic image suffices: the warmup only exercises code paths,
            # pixel content is irrelevant
            _pil_img = _PILImage.new("RGB", (224, 224), (128, 128, 128))
            _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            assert max_pixels is not None
            assert min_pixels is not None

            # sample A: with image
            _msgs_a = [{"role": "user", "content": [
                {"type": "image", "image": _pil_img},
                {"type": "text", "text": "hello"},
            ]}]
            _text_a = _processor.apply_chat_template(
                _msgs_a, tokenize=False, add_generation_prompt=False,
            )
            _result_a = _processor(
                text=[_text_a], images=[_pil_img], return_tensors="pt",
                max_pixels=max_pixels, min_pixels=min_pixels,
            )
            _ids_a = _result_a["input_ids"][0].tolist()

            # sample B: text only
            _msgs_b = [{"role": "user", "content": [
                {"type": "text", "text": "hello"},
            ]}]
            _text_b = _processor.apply_chat_template(
                _msgs_b, tokenize=False, add_generation_prompt=False,
            )
            _result_b = _processor(text=[_text_b], return_tensors="pt")
            _ids_b = _result_b["input_ids"][0].tolist()

            # image_data two-level structure: matches the training collate output
            # ([[Image], [None]] -> sglang recognizes modalities=["image", None])
            self.engine.generate(
                input_ids=[_ids_a, _ids_b],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_hidden_states=True,
                image_data=[[_pil_img], [None]],
                # NOTE: image_data=None means text-only to sglang (all samples go
                # through text preprocessing); with image_data=list, sglang first
                # checks image_data[0] -- if it is not a list, every element gets
                # wrapped in an extra list -- then preprocesses per sample:
                # [None] takes the text path, [img1, [img2, ...]] the image-text path
            )
        else:
            self.engine.generate(
                input_ids=[[0, 1, 2, 3]],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_hidden_states=True,
            )

        # ── 6. Print engine info ─────────────────────────────────────
        import sys

        _server_args = getattr(self.engine, "server_args", None)
        _info_parts = [
            f"model={model_path}",
            f"tp={tp_size}",
            f"context_length={max_model_len}",
            f"layers={self._captured_indices}",
            f"chunked_prefill_size={getattr(_server_args, 'chunked_prefill_size', 'N/A')}",
            f"max_prefill_tokens={getattr(_server_args, 'max_prefill_tokens', 'N/A')}",
            f"disable_radix_cache={getattr(_server_args, 'disable_radix_cache', 'N/A')}",
        ]
        print(
            f"[SGLangModel] Ready: {' | '.join(_info_parts)}",
            file=sys.stderr,
            flush=True,
        )

    # ─── Properties ──────────────────────────────────────────────────

    @property
    def layer_indices(self) -> List[int]:
        """Original layer indices that are captured (e.g. [0, 1, ..., N])."""
        return list(self._captured_indices)

    @property
    def num_captured_layers(self) -> int:
        return len(self._captured_indices)

    # ─── Forward ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Union[List[int], List[List[int]], torch.Tensor],
        output_hidden_states: bool = True,
        image_data: Optional[Union[List, List[List]]] = None,
    ) -> SGLangModelOutput:
        """Run prefill on one sequence or a batch.

        image_data contains per-sample [images] or None. Ragged input is accepted,
        but stacked hidden-state outputs require equal padded lengths. Returns a
        sequence of [batch, seq_len, hidden_size] tensors when requested; tensor
        inputs also place these outputs on the input device.
        """
        # ── Normalise input_ids → List[List[int]] ────────────────────
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids_list = [input_ids.tolist()]
            else:
                input_ids_list = input_ids.tolist()
        elif isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], int):
            input_ids_list = [input_ids]
        else:
            input_ids_list = input_ids

        batch_size = len(input_ids_list)
        self._forward_count += 1

        # strip pads internally so that, without an attention mask on the SGLang
        # side, valid tokens are never polluted by pads and RoPE positions stay
        # correct; the output side pads zeros back per padding_side to keep the
        # input shape.
        pad_id = self.pad_token_id
        stripped_ids_list: List[List[int]] = []
        n_pads: List[int] = []
        for ids in input_ids_list:
            if self.padding_side == "left":
                k = 0
                while k < len(ids) and ids[k] == pad_id:
                    k += 1
                stripped_ids_list.append(list(ids[k:]))
                n_pads.append(k)
            else:  # right
                k = len(ids)
                while k > 0 and ids[k - 1] == pad_id:
                    k -= 1
                stripped_ids_list.append(list(ids[:k]))
                n_pads.append(len(ids) - k)
        stripped_lens = [len(ids) for ids in stripped_ids_list]

        # SGLang decides the batch format by whether image_data[0] is a list
        # (io_struct.py _normalize_image_data); a leading None takes the wrong
        # branch and wraps every entry in an extra layer, turning image samples
        # into [[PIL]] and raising "Invalid image". Replace bare None with [None]
        # (SGLang explicitly recognizes [None] as no image) to protect
        # text-first-image-later batches.
        if (
            image_data is not None
            and isinstance(image_data, list)
            and any(isinstance(x, list) for x in image_data)
        ):
            image_data = [[None] if x is None else x for x in image_data]

        try:
            outputs = self.engine.generate(
                input_ids=stripped_ids_list,
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_hidden_states=output_hidden_states,
                image_data=image_data,
            )
        except Exception as e:
            self._mismatch_count += 1
            rate = self._mismatch_count / self._forward_count
            raise RuntimeError(
                f"[SGLangModel] {e}\n"
                f"  batch_size={batch_size}\n"
                f"  engine failure rate so far: {self._mismatch_count}/{self._forward_count} = {rate:.2%}\n"
            )

        if not output_hidden_states:
            return SGLangModelOutput()

        # ── Parse hidden states ──────────────────────────────────────
        per_sample: List[torch.Tensor] = []  # each [num_repr, seq_len_i, hidden]
        for out in outputs:
            raw_hs = out["meta_info"]["hidden_states"][0]  # prefill step
            if isinstance(raw_hs, torch.Tensor):
                hs = raw_hs.float()
            else:
                hs = torch.tensor(raw_hs, dtype=torch.float32)

            seq_len = hs.shape[0]
            num_repr = hs.shape[1] // self.hidden_size
            # [seq_len, num_repr, hidden] -> [num_repr, seq_len, hidden]
            per_sample.append(
                hs.reshape(seq_len, num_repr, self.hidden_size).permute(1, 0, 2)
            )

        num_repr = per_sample[0].shape[0]
        seq_lens = [t.shape[1] for t in per_sample]
        input_lens = stripped_lens  # the engine-side expected lengths are the stripped ones

        # validate: every output seq_len must equal the stripped input_len
        if seq_lens != input_lens:
            self._mismatch_count += 1
            rate = self._mismatch_count / self._forward_count
            raise RuntimeError(
                f"[SGLangModel] hidden_states length mismatch with input\n"
                f"  batch_size={batch_size}\n"
                f"  input_lens (stripped)={input_lens}\n"
                f"  output_seq_lens={seq_lens}\n"
                f"  diff={[s - i for s, i in zip(seq_lens, input_lens)]}\n"
                f"  mismatch rate so far: {self._mismatch_count}/{self._forward_count} = {rate:.2%}\n"
                f"possible cause 1: prefill chunking; set chunked_prefill_size=-1 and ensure max_prefill_tokens covers the batch token total.\n"
                f"possible cause 2: VL image-preprocessing mismatch; check model_path/preprocessor_config.json and keep it consistent with preprocessing"
            )

        # ── Pad back to the original padded_len: concat zero vectors along the seq dim per padding_side ──
        if any(n > 0 for n in n_pads):
            padded_per_sample: List[torch.Tensor] = []
            for b, t in enumerate(per_sample):
                n_pad = n_pads[b]
                if n_pad == 0:
                    padded_per_sample.append(t)
                    continue
                pad_block = torch.zeros(
                    (t.shape[0], n_pad, t.shape[2]),
                    dtype=t.dtype, device=t.device,
                )
                if self.padding_side == "left":
                    padded_per_sample.append(torch.cat([pad_block, t], dim=1))
                else:
                    padded_per_sample.append(torch.cat([t, pad_block], dim=1))
            per_sample = padded_per_sample

        # Same length (padded_lens) → stack into [batch, seq_len, hidden] per layer
        hidden_states = tuple(
            torch.stack([per_sample[b][j] for b in range(batch_size)])
            for j in range(num_repr)
        )

        # ------- move to the input_ids device -------
        if isinstance(input_ids, torch.Tensor):
            hidden_states = [hs.to(input_ids.device) for hs in hidden_states]

        return SGLangModelOutput(hidden_states=hidden_states)

    def __call__(
        self,
        input_ids: Union[List[int], List[List[int]], torch.Tensor, None] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> SGLangModelOutput:
        """Callable interface matching ``model(input_ids=..., output_hidden_states=...)``."""
        if input_ids is None:
            raise ValueError("input_ids is required")
        return self.forward(
            input_ids,
            output_hidden_states=output_hidden_states,
            image_data=kwargs.pop('image_data', None),
        )

    # ─── Lifecycle ───────────────────────────────────────────────────

    def shutdown(self):
        """Shut down the engine.  Safe to call multiple times."""
        if getattr(self, "engine", None) is not None:
            self.engine.shutdown()
            self.engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    def __repr__(self):
        status = "active" if self.engine is not None else "shutdown"
        return (
            f"SGLangModel(model={self.model_path!r}, "
            f"layers={self._captured_indices}, status={status})"
        )

    # ─── VL preprocessor config alignment ────────────────────────────

    def _patch_preprocessor_config(self, model_path, max_pixels: int, min_pixels: int):
        """Align preprocessor_config.json image resolution with the training
        pipeline (delegates to the module-level function)."""
        patch_preprocessor_config_file(model_path, max_pixels, min_pixels)
