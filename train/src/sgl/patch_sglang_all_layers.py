#!/usr/bin/env python3
"""Patch supported SGLang sources to capture intermediate hidden states.

Call auto_patch() before importing sglang. Direct source changes are visible to
subprocesses and remain installed until explicitly restored. Install on every
node unless the filesystem is shared.

CLI: python patch_sglang_all_layers.py --install|--uninstall|--status

TP and independent DP ranks are supported. PP is rejected because non-final
ranks do not return the auxiliary hidden states required by this extractor.
"""

import argparse
import hashlib
import importlib.util
import os
import shutil
import sys

BACKUP_SUFFIX = ".bak.before_all_hidden_patch"
_PATCH_MARKER_BASE = "PATCHED-ALL-HIDDEN-CAPTURE"
_ENV_PATCHED = "_SGLANG_ALL_HIDDEN_PATCHED"
_ENV_CAPTURE_LAYERS = "SGLANG_CAPTURE_LAYERS"

# Tested and supported sglang versions.  The patch relies on exact source
# code patterns; other versions may have different code and silently break.
SUPPORTED_VERSIONS = ["0.5.10", "0.5.13"]

# Detected sglang version (set by _check_version())
_detected_version = None

# Supported HuggingFace config.model_type values; each maps to registered patches.
SUPPORTED_MODEL_TYPES = [
    "qwen3", "qwen3_moe",
    "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe",
    "glm4v_moe",
    "deepseek_v4",  # 0.5.13+ only
]

# ── Load-time banner (inserted into source files) ──────────────────────
_BANNER_LINE = (
    'print("\\n" + "="*70 + "\\n[PATCH-ALL-HIDDEN] " + __file__'
    ' + "\\n  -> ALL-LAYER HIDDEN STATES CAPTURE ENABLED\\n"'
    ' + "="*70, file=__import__("sys").stderr, flush=True)'
)


# =====================================================================
# Version check
# =====================================================================

def _get_sglang_version():
    """Prefer package metadata and fall back to importing sglang."""
    # Method 1: package metadata (pip-installed)
    try:
        from importlib.metadata import version
        return version("sglang")
    except Exception:
        pass
    # Method 2: import sglang.__version__
    try:
        import sglang
        return getattr(sglang, "__version__", None)
    except ImportError:
        pass
    return None


def _normalize_version(ver: str) -> str:
    """Strip .postN / .devN / +local suffixes → keep only X.Y.Z."""
    for sep in ("+", ".post", ".dev", "a", "b", "rc"):
        ver = ver.split(sep)[0]
    return ver


def _check_version():
    """Verify sglang version is in SUPPORTED_VERSIONS; return normalised base version."""
    global _detected_version
    raw = _get_sglang_version()
    if raw is None:
        raise RuntimeError(
            "Cannot determine sglang version. Is sglang installed?"
        )
    base = _normalize_version(raw)
    if base not in SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"sglang version {raw!r} (base={base!r}) is NOT supported by this patch.\n"
            f"Supported versions: {SUPPORTED_VERSIONS}\n"
            f"The patch modifies sglang source code and is version-specific.\n"
            f"Using it on an untested version may silently produce wrong results."
        )
    _detected_version = base
    print(f"[PATCH] sglang version {raw} (base={base}) \u2713 (supported)", file=sys.stderr, flush=True)
    return base


def load_hf_config(model_path, **kwargs):
    """Load a model config; novel architectures not yet built into transformers
    (e.g. deepseek_v4) fall back to the generic PretrainedConfig."""
    from transformers import AutoConfig
    try:
        return AutoConfig.from_pretrained(model_path, **kwargs)
    except Exception:
        from transformers import PretrainedConfig
        return PretrainedConfig.from_pretrained(model_path, **kwargs)


# The official DeepSeek-V4 release ships without an HF chat_template (the
# official conversation encoding lives in encoding/encoding_dsv4.py of the base
# repo). Below is an equivalent jinja template assembled from the official doc
# format: <bos>{system}<User>{user}<Assistant>{response}<eos>...
# sample assistant contents already carry a full end-of-turn block, so the
# template concatenates verbatim; add_bos_token/add_eos_token are both False,
# so the template's explicit BOS is never duplicated.
DSV4_CHAT_TEMPLATE = (
    "<｜begin▁of▁sentence｜>"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}"
    "{% elif message['role'] == 'user' %}<｜User｜>{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}<｜Assistant｜>{{ message['content'] }}<｜end▁of▁sentence｜>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<｜Assistant｜>{% endif %}"
)


def ensure_dsv4_chat_template(tokenizer, model_path):
    """Inject the official-format chat_template for deepseek_v4 models
    (idempotent: an existing template is left untouched)."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    try:
        model_type = getattr(load_hf_config(model_path, trust_remote_code=True), "model_type", None)
    except Exception:
        model_type = None
    if model_type == "deepseek_v4":
        tokenizer.chat_template = DSV4_CHAT_TEMPLATE
        print("[PATCH] deepseek_v4: injected official-format chat_template", file=sys.stderr, flush=True)


def check_model_support(model_path):
    """Verify that the model architecture is supported by the patch.

    Raises RuntimeError if ``config.model_type`` is not in
    :data:`SUPPORTED_MODEL_TYPES`.
    """
    config = load_hf_config(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise RuntimeError(
            f"Model architecture {model_type!r} (from {model_path}) is not supported.\n"
            f"Supported model types: {SUPPORTED_MODEL_TYPES}\n"
            f"The patch modifies architecture-specific source files."
        )
    print(f"[PATCH] model type {model_type!r} \u2713 (supported)", file=sys.stderr, flush=True)


# =====================================================================
# Auto-detect sglang installation
# =====================================================================

def find_sglang_models_dir():
    """Find sglang/srt/models/ in the current Python environment."""
    # Method 1: importlib (no side-effects, does not execute sglang)
    try:
        spec = importlib.util.find_spec("sglang.srt.models")
        if spec and spec.submodule_search_locations:
            d = spec.submodule_search_locations[0]
            if os.path.isfile(os.path.join(d, "qwen2.py")):
                return d
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    # Method 2: scan sys.path
    for p in sys.path:
        candidate = os.path.join(p, "sglang", "srt", "models")
        if os.path.isfile(os.path.join(candidate, "qwen2.py")):
            return candidate

    raise RuntimeError(
        "Cannot find sglang installation (sglang.srt.models.qwen2).\n"
        "Is sglang installed?  Alternatively pass --models-dir explicitly."
    )


# =====================================================================
# Patch definitions
# =====================================================================
# Each patch is (find_text, replace_text). find_text MUST appear exactly
# once in the file; otherwise the script aborts.
#
# Functional patches are separated from banner patches so that we can
# compute a content-based digest WITHOUT circularity (the digest itself
# is embedded in the banner).

def _qwen3_functional_patches(version):
    """Patch 2: actual code modifications for qwen3.py."""
    return [
        # 2. Always enable capture_aux_hidden_states in Qwen3ForCausalLM
        (
            '        # For EAGLE3 support\n'
            '        self.capture_aux_hidden_states = False',
            # ↓↓↓
            '        # PATCHED: always capture aux hidden states (original: False)\n'
            '        self.capture_aux_hidden_states = True',
        ),
    ]


def _qwen3_5_functional_patches(version):
    """Patches for qwen3_5.py (Qwen3.5 dense + MoE, 0.5.10+ only)."""
    return [
        # 1. __init__: configurable layer capture + PP rejection
        (
            '        self.layers, self._start_layer, self._end_layer = make_layers(\n'
            '            config.num_hidden_layers,\n'
            '            get_layer,\n'
            '            pp_rank=self.pp_group.rank_in_group,\n'
            '            pp_size=self.pp_group.world_size,\n'
            '            prefix=f"{prefix}.layers",\n'
            '        )\n'
            '\n'
            '        # Final normalization\n'
            '        if self.pp_group.is_last_rank:\n'
            '            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n'
            '        else:\n'
            '            self.norm = PPMissingLayer()',
            # ↓↓↓
            '        self.layers, self._start_layer, self._end_layer = make_layers(\n'
            '            config.num_hidden_layers,\n'
            '            get_layer,\n'
            '            pp_rank=self.pp_group.rank_in_group,\n'
            '            pp_size=self.pp_group.world_size,\n'
            '            prefix=f"{prefix}.layers",\n'
            '        )\n'
            '\n'
            '        # PATCHED: configurable layer capture (env SGLANG_CAPTURE_LAYERS)\n'
            '        _cap = __import__("os").environ.get("SGLANG_CAPTURE_LAYERS", "all")\n'
            '        if _cap.strip().lower() == "all":\n'
            '            self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '            self._capture_final_norm = True\n'
            '        else:\n'
            '            _req = {int(x) for x in _cap.split(",") if x.strip().isdigit()}\n'
            '            _valid = set(range(self.start_layer, self.end_layer))\n'
            '            self.layers_to_capture = sorted(_req & _valid)\n'
            '            self._capture_final_norm = self.end_layer in _req\n'
            '            if not self.layers_to_capture and not self._capture_final_norm:\n'
            '                self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '                self._capture_final_norm = True\n'
            '        # PATCHED: reject pipeline parallelism (aux_hidden_states lost between PP stages)\n'
            '        if self.pp_group.world_size > 1:\n'
            '            raise RuntimeError(\n'
            '                "[PATCH] Pipeline parallelism (pp_size>1) is NOT supported for "\n'
            '                "hidden states extraction. Non-last PP ranks discard "\n'
            '                "aux_hidden_states during PPProxyTensors transfer. "\n'
            '                "Use tensor parallelism (tp_size) instead."\n'
            '            )\n'
            '\n'
            '        # Final normalization\n'
            '        if self.pp_group.is_last_rank:\n'
            '            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n'
            '        else:\n'
            '            self.norm = PPMissingLayer()',
        ),
        # 2. forward: pre-allocated buffer capture
        (
            '        # Pass through decoder layers\n'
            '        for layer_idx in range(self.start_layer, self.end_layer):\n'
            '            layer = self.layers[layer_idx]\n'
            '            with get_global_expert_distribution_recorder().with_current_layer(\n'
            '                layer_idx\n'
            '            ):\n'
            '                hidden_states, residual = layer(\n'
            '                    positions=positions,\n'
            '                    hidden_states=hidden_states,\n'
            '                    residual=residual,\n'
            '                    forward_batch=forward_batch,\n'
            '                )\n'
            '\n'
            '            # Process deepstack embeddings if provided\n'
            '            if (\n'
            '                input_deepstack_embeds is not None\n'
            '                and input_deepstack_embeds.numel() > 0\n'
            '                and layer_idx < 3\n'
            '            ):\n'
            '                sep = self.hidden_size * layer_idx\n'
            '                hidden_states.add_(\n'
            '                    input_deepstack_embeds[:, sep : sep + self.hidden_size]\n'
            '                )\n'
            '\n'
            '        # Return intermediate tensors for pipeline parallelism\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '\n'
            '        # Apply final normalization\n'
            '        if hidden_states.shape[0] != 0:\n'
            '            if residual is None:\n'
            '                hidden_states = self.norm(hidden_states)\n'
            '            else:\n'
            '                hidden_states, _ = self.norm(hidden_states, residual)\n'
            '\n'
            '        return hidden_states',
            # ↓↓↓
            '        # PATCHED: pre-allocate contiguous buffer (eliminates per-layer alloc + torch.cat)\n'
            '        _n_cap = len(self.layers_to_capture) + (1 if self._capture_final_norm else 0)\n'
            '        _aux_buf = torch.empty(\n'
            '            hidden_states.shape[0], _n_cap, hidden_states.shape[-1],\n'
            '            device=hidden_states.device, dtype=hidden_states.dtype,\n'
            '        ) if _n_cap > 0 else None\n'
            '        _aux_idx = 0\n'
            '        # Pass through decoder layers\n'
            '        for layer_idx in range(self.start_layer, self.end_layer):\n'
            '            if layer_idx in self.layers_to_capture:\n'
            '                # PATCHED: write into buffer slice (copy semantics)\n'
            '                _aux_buf[:, _aux_idx] = (\n'
            '                    hidden_states + residual if residual is not None else hidden_states\n'
            '                )\n'
            '                _aux_idx += 1\n'
            '            layer = self.layers[layer_idx]\n'
            '            with get_global_expert_distribution_recorder().with_current_layer(\n'
            '                layer_idx\n'
            '            ):\n'
            '                hidden_states, residual = layer(\n'
            '                    positions=positions,\n'
            '                    hidden_states=hidden_states,\n'
            '                    residual=residual,\n'
            '                    forward_batch=forward_batch,\n'
            '                )\n'
            '\n'
            '            # Process deepstack embeddings if provided\n'
            '            if (\n'
            '                input_deepstack_embeds is not None\n'
            '                and input_deepstack_embeds.numel() > 0\n'
            '                and layer_idx < 3\n'
            '            ):\n'
            '                sep = self.hidden_size * layer_idx\n'
            '                hidden_states.add_(\n'
            '                    input_deepstack_embeds[:, sep : sep + self.hidden_size]\n'
            '                )\n'
            '\n'
            '        # Return intermediate tensors for pipeline parallelism\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '\n'
            '        # Apply final normalization\n'
            '        if hidden_states.shape[0] != 0:\n'
            '            if residual is None:\n'
            '                hidden_states = self.norm(hidden_states)\n'
            '            else:\n'
            '                hidden_states, _ = self.norm(hidden_states, residual)\n'
            '\n'
            '        # PATCHED: append final RMSNorm + return pre-concatenated single tensor\n'
            '        if self._capture_final_norm and _aux_buf is not None:\n'
            '            _aux_buf[:, _aux_idx] = hidden_states\n'
            '            _aux_idx += 1\n'
            '        if _aux_buf is not None and _aux_idx > 0:\n'
            '            return hidden_states, [_aux_buf[:, :_aux_idx].reshape(hidden_states.shape[0], -1)]\n'
            '\n'
            '        return hidden_states',
        ),
    ]


def qwen3_5_patches(version):
    """Patches for qwen3_5.py (0.5.10+ only)."""
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _qwen3_5_functional_patches(version)


def _qwen3_vl_functional_patches(version):
    """Patches for qwen3_vl.py: backbone buffer capture + VL wrapper aux passthrough (0.5.10+ only)."""
    return [
        # 1. Qwen3LLMModel.forward(): pre-allocated buffer replaces list append
        #    Preserves all deepstack logic; adds _capture_final_norm support.
        (
            '        aux_hidden_states = []\n'
            '        for layer_idx, layer in enumerate(\n'
            '            self.layers[self.start_layer : self.end_layer]\n'
            '        ):\n'
            '            layer_idx = layer_idx + self.start_layer\n'
            '            if layer_idx in self.layers_to_capture:\n'
            '                aux_hidden_states.append(\n'
            '                    hidden_states + residual if residual is not None else hidden_states\n'
            '                )\n'
            '\n'
            '            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.\n'
            '            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549\n'
            '            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack\n'
            '            # The order matters because addition with different tensors is not associative in practice.\n'
            '            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.\n'
            '            deepstack_embeds = self.get_deepstack_embeds(\n'
            '                layer_idx - 1, input_deepstack_embeds\n'
            '            )\n'
            '            hidden_states, residual = layer(\n'
            '                positions,\n'
            '                hidden_states,\n'
            '                forward_batch,\n'
            '                residual,\n'
            '                post_residual_addition=deepstack_embeds,\n'
            '            )\n'
            '\n'
            '        # Handle deepstack for the last processed layer if it exists.\n'
            '        last_deepstack = self.get_deepstack_embeds(\n'
            '            self.end_layer - 1, input_deepstack_embeds\n'
            '        )\n'
            '\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '        else:\n'
            '            if hidden_states.shape[0] != 0:\n'
            '                if residual is None:\n'
            '                    hidden_states = self.norm(hidden_states)\n'
            '                else:\n'
            '                    hidden_states, _ = self.norm(\n'
            '                        hidden_states, residual, post_residual_addition=last_deepstack\n'
            '                    )\n'
            '\n'
            '        if len(aux_hidden_states) == 0:\n'
            '            return hidden_states\n'
            '\n'
            '        return hidden_states, aux_hidden_states',
            # ↓↓↓
            '        # PATCHED: pre-allocate contiguous buffer (eliminates per-layer alloc + torch.cat)\n'
            '        _n_cap = len(self.layers_to_capture) + (1 if self._capture_final_norm else 0)\n'
            '        _aux_buf = torch.empty(\n'
            '            hidden_states.shape[0], _n_cap, hidden_states.shape[-1],\n'
            '            device=hidden_states.device, dtype=hidden_states.dtype,\n'
            '        ) if _n_cap > 0 else None\n'
            '        _aux_idx = 0\n'
            '        for layer_idx, layer in enumerate(\n'
            '            self.layers[self.start_layer : self.end_layer]\n'
            '        ):\n'
            '            layer_idx = layer_idx + self.start_layer\n'
            '            if layer_idx in self.layers_to_capture:\n'
            '                # PATCHED: write into buffer slice (copy semantics)\n'
            '                _aux_buf[:, _aux_idx] = (\n'
            '                    hidden_states + residual if residual is not None else hidden_states\n'
            '                )\n'
            '                _aux_idx += 1\n'
            '\n'
            '            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.\n'
            '            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549\n'
            '            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack\n'
            '            # The order matters because addition with different tensors is not associative in practice.\n'
            '            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.\n'
            '            deepstack_embeds = self.get_deepstack_embeds(\n'
            '                layer_idx - 1, input_deepstack_embeds\n'
            '            )\n'
            '            hidden_states, residual = layer(\n'
            '                positions,\n'
            '                hidden_states,\n'
            '                forward_batch,\n'
            '                residual,\n'
            '                post_residual_addition=deepstack_embeds,\n'
            '            )\n'
            '\n'
            '        # Handle deepstack for the last processed layer if it exists.\n'
            '        last_deepstack = self.get_deepstack_embeds(\n'
            '            self.end_layer - 1, input_deepstack_embeds\n'
            '        )\n'
            '\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '        else:\n'
            '            if hidden_states.shape[0] != 0:\n'
            '                if residual is None:\n'
            '                    hidden_states = self.norm(hidden_states)\n'
            '                else:\n'
            '                    hidden_states, _ = self.norm(\n'
            '                        hidden_states, residual, post_residual_addition=last_deepstack\n'
            '                    )\n'
            '\n'
            '        # PATCHED: append final RMSNorm + return pre-concatenated single tensor\n'
            '        if self._capture_final_norm and _aux_buf is not None:\n'
            '            _aux_buf[:, _aux_idx] = hidden_states\n'
            '            _aux_idx += 1\n'
            '        if _aux_buf is not None and _aux_idx > 0:\n'
            '            return hidden_states, [_aux_buf[:, :_aux_idx].reshape(hidden_states.shape[0], -1)]\n'
            '\n'
            '        return hidden_states',
        ),
        # 2. Qwen3VLForConditionalGeneration.forward(): unpack aux_hidden_states and pass to logits_processor
        (
            '        hidden_states = general_mm_embed_routine(\n'
            '            input_ids=input_ids,\n'
            '            forward_batch=forward_batch,\n'
            '            language_model=self.model,\n'
            '            multimodal_model=self,\n'
            '            positions=positions,\n'
            '            use_deepstack=self.use_deepstack,\n'
            '            pp_proxy_tensors=pp_proxy_tensors,\n'
            '        )\n'
            '\n'
            '        if self.pp_group.is_last_rank:\n'
            '            if not get_embedding:\n'
            '                return self.logits_processor(\n'
            '                    input_ids,\n'
            '                    hidden_states,\n'
            '                    self.lm_head,\n'
            '                    forward_batch,\n'
            '                )',
            # ↓↓↓
            '        hidden_states = general_mm_embed_routine(\n'
            '            input_ids=input_ids,\n'
            '            forward_batch=forward_batch,\n'
            '            language_model=self.model,\n'
            '            multimodal_model=self,\n'
            '            positions=positions,\n'
            '            use_deepstack=self.use_deepstack,\n'
            '            pp_proxy_tensors=pp_proxy_tensors,\n'
            '        )\n'
            '\n'
            '        # PATCHED: unpack aux_hidden_states from backbone\n'
            '        aux_hidden_states = None\n'
            '        if isinstance(hidden_states, tuple):\n'
            '            hidden_states, aux_hidden_states = hidden_states\n'
            '\n'
            '        if self.pp_group.is_last_rank:\n'
            '            if not get_embedding:\n'
            '                return self.logits_processor(\n'
            '                    input_ids,\n'
            '                    hidden_states,\n'
            '                    self.lm_head,\n'
            '                    forward_batch,\n'
            '                    aux_hidden_states,\n'
            '                )',
        ),
    ]


def qwen3_vl_patches(version):
    """Patches for qwen3_vl.py (0.5.10+ only)."""
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _qwen3_vl_functional_patches(version)


def _qwen3_vl_moe_functional_patches(version):
    """Patches for qwen3_vl_moe.py: Qwen3MoeLLMModel.forward() buffer capture.

    Notes:
      - Qwen3VLMoeForConditionalGeneration inherits Qwen3VLForConditionalGeneration
        without overriding forward(), so the aux_hidden_states unpack patch in
        qwen3_vl.py takes effect via inheritance (no additional wrapper patch).
      - Qwen3MoeLLMModel.__init__ relies on Qwen2MoeModel.__init__ (already
        patched to configure layers_to_capture + _capture_final_norm).
      - The backbone forward code is identical to qwen3_vl.py's Qwen3LLMModel
        except for ``layer_idx += self.start_layer`` (compound assignment) vs
        ``layer_idx = layer_idx + self.start_layer`` in qwen3_vl.py.
    """
    return [
        # 1. Qwen3MoeLLMModel.forward(): pre-allocated buffer replaces list append
        (
            '        aux_hidden_states = []\n'
            '        for layer_idx, layer in enumerate(\n'
            '            self.layers[self.start_layer : self.end_layer]\n'
            '        ):\n'
            '            layer_idx += self.start_layer\n'
            '            if layer_idx in self.layers_to_capture:\n'
            '                aux_hidden_states.append(\n'
            '                    hidden_states + residual if residual is not None else hidden_states\n'
            '                )\n'
            '\n'
            '            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.\n'
            '            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549\n'
            '            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack\n'
            '            # The order matters because addition with different tensors is not associative in practice.\n'
            '            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.\n'
            '            deepstack_embeds = self.get_deepstack_embeds(\n'
            '                layer_idx - 1, input_deepstack_embeds\n'
            '            )\n'
            '            hidden_states, residual = layer(\n'
            '                positions,\n'
            '                hidden_states,\n'
            '                forward_batch,\n'
            '                residual,\n'
            '                post_residual_addition=deepstack_embeds,\n'
            '            )\n'
            '\n'
            '        # Handle deepstack for the last processed layer if it exists.\n'
            '        last_deepstack = self.get_deepstack_embeds(\n'
            '            self.end_layer - 1, input_deepstack_embeds\n'
            '        )\n'
            '\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '        else:\n'
            '            if hidden_states.shape[0] != 0:\n'
            '                if residual is None:\n'
            '                    hidden_states = self.norm(hidden_states)\n'
            '                else:\n'
            '                    hidden_states, _ = self.norm(\n'
            '                        hidden_states, residual, post_residual_addition=last_deepstack\n'
            '                    )\n'
            '\n'
            '        if len(aux_hidden_states) == 0:\n'
            '            return hidden_states\n'
            '\n'
            '        return hidden_states, aux_hidden_states',
            # ↓↓↓
            '        # PATCHED: pre-allocate contiguous buffer (eliminates per-layer alloc + torch.cat)\n'
            '        _n_cap = len(self.layers_to_capture) + (1 if self._capture_final_norm else 0)\n'
            '        _aux_buf = torch.empty(\n'
            '            hidden_states.shape[0], _n_cap, hidden_states.shape[-1],\n'
            '            device=hidden_states.device, dtype=hidden_states.dtype,\n'
            '        ) if _n_cap > 0 else None\n'
            '        _aux_idx = 0\n'
            '        for layer_idx, layer in enumerate(\n'
            '            self.layers[self.start_layer : self.end_layer]\n'
            '        ):\n'
            '            layer_idx += self.start_layer\n'
            '            if layer_idx in self.layers_to_capture:\n'
            '                # PATCHED: write into buffer slice (copy semantics)\n'
            '                _aux_buf[:, _aux_idx] = (\n'
            '                    hidden_states + residual if residual is not None else hidden_states\n'
            '                )\n'
            '                _aux_idx += 1\n'
            '\n'
            '            # SGLang applies residual at the START of the next layer, not at the END like HuggingFace.\n'
            '            # See: https://github.com/huggingface/transformers/blob/v5.0.0rc0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L549\n'
            '            # To match HF behavior, deepstack must be added AFTER residual: (hidden_states + residual) + deepstack\n'
            '            # The order matters because addition with different tensors is not associative in practice.\n'
            '            # Deepstack for prev_layer is applied at the start of current layer via post_residual_addition.\n'
            '            deepstack_embeds = self.get_deepstack_embeds(\n'
            '                layer_idx - 1, input_deepstack_embeds\n'
            '            )\n'
            '            hidden_states, residual = layer(\n'
            '                positions,\n'
            '                hidden_states,\n'
            '                forward_batch,\n'
            '                residual,\n'
            '                post_residual_addition=deepstack_embeds,\n'
            '            )\n'
            '\n'
            '        # Handle deepstack for the last processed layer if it exists.\n'
            '        last_deepstack = self.get_deepstack_embeds(\n'
            '            self.end_layer - 1, input_deepstack_embeds\n'
            '        )\n'
            '\n'
            '        if not self.pp_group.is_last_rank:\n'
            '            return PPProxyTensors(\n'
            '                {\n'
            '                    "hidden_states": hidden_states,\n'
            '                    "residual": residual,\n'
            '                }\n'
            '            )\n'
            '        else:\n'
            '            if hidden_states.shape[0] != 0:\n'
            '                if residual is None:\n'
            '                    hidden_states = self.norm(hidden_states)\n'
            '                else:\n'
            '                    hidden_states, _ = self.norm(\n'
            '                        hidden_states, residual, post_residual_addition=last_deepstack\n'
            '                    )\n'
            '\n'
            '        # PATCHED: append final RMSNorm + return pre-concatenated single tensor\n'
            '        if self._capture_final_norm and _aux_buf is not None:\n'
            '            _aux_buf[:, _aux_idx] = hidden_states\n'
            '            _aux_idx += 1\n'
            '        if _aux_buf is not None and _aux_idx > 0:\n'
            '            return hidden_states, [_aux_buf[:, :_aux_idx].reshape(hidden_states.shape[0], -1)]\n'
            '\n'
            '        return hidden_states',
        ),
    ]


def qwen3_vl_moe_patches(version):
    """Patches for qwen3_vl_moe.py (0.5.10+ only)."""
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _qwen3_vl_moe_functional_patches(version)


def _qwen_vl_processor_functional_patches(version):
    """Patches for multimodal/processors/qwen_vl.py.

    Force SGLang to use LEGACY PATH in `load_mm_data` by setting
    `support_dynamic_frame_expansion = True` on QwenVLImageProcessor.

    Why:
      Qwen-VL's `image_token_regex` matches the entire vision block
      (<|vision_start|>...<|vision_end|>) as ONE occurrence, so
      `cnt[IMAGE] == n_image` → FAST PATH, which hands the decoded text
      (containing N expanded <|image_pad|> tokens) directly to the HF
      processor. HF's Qwen3VLProcessor.__call__ then iterates *individual*
      <|image_pad|> tokens and indexes into `image_grid_thw` once per
      occurrence → IndexError on the 2nd one (only 1 image provided).

      LEGACY PATH rebuilds the text with a single <|image_pad|> per image,
      matching what `apply_chat_template(tokenize=False)` produces in the
      training pipeline. This guarantees the re-tokenized input_ids exactly
      equal the original, so `seq_len` aligns with no content drift.
    """
    return [
        (
            '# Compatible with Qwen-VL & Qwen-Omni Series\n'
            'class QwenVLImageProcessor(SGLangBaseProcessor):\n'
            '    supports_transformers_backend = True',
            # ↓↓↓
            '# Compatible with Qwen-VL & Qwen-Omni Series\n'
            'class QwenVLImageProcessor(SGLangBaseProcessor):\n'
            '    # PATCHED: force LEGACY PATH in base_processor.load_mm_data\n'
            '    # so HF processor receives a single <|image_pad|> per image\n'
            '    # (aligned with apply_chat_template output) → seq_len stays consistent.\n'
            '    support_dynamic_frame_expansion = True\n'
            '    supports_transformers_backend = True',
        ),
    ]


def qwen_vl_processor_patches(version):
    """Patches for multimodal/processors/qwen_vl.py (0.5.10+ only)."""
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal processor code)
        (
            'import math\n'
            'import os\n'
            'import re',
            # ↓↓↓
            f'# {marker}\n'
            'import math\n'
            'import os\n'
            'import re',
        ),
    ] + _qwen_vl_processor_functional_patches(version)


def _glm4_moe_functional_patches(version):
    """Patches for glm4_moe.py: Glm4MoeModel backbone capture (0.5.10+ only).

    The 0.5.10 Glm4MoeModel.forward already has an EAGLE3 capture loop, but:
      (a) layers_to_capture is always empty (nobody enables it);
      (b) the official capture writes `hidden_states + residual`, which crashes
          at layer 0 where residual=None (official EAGLE3 only captures
          [2, n//2, n-3] and never layer 0, so it was never exposed);
      (c) the final RMSNorm is not captured.
    This aligns with the qwen family: env-driven layer selection + a
    pre-allocated buffer + the final norm appended.
    Also: Glm4MoeForCausalLM.capture_aux_hidden_states is set True so the
    glm4_moe text-only architecture stays self-consistent under this patch (the
    backbone always returns a tuple that must be unpacked).
    """
    return [
        # 1. Configurable layer capture via env SGLANG_CAPTURE_LAYERS
        #    "all" (default) = every layer + final norm
        #    Index semantics (N = num_hidden_layers):
        #      0     = embedding,  1..N-1 = after transformer layers,  N = final RMSNorm
        (
            '        self.layers_to_capture = []',
            # ↓↓↓
            '        # PATCHED: configurable layer capture (env SGLANG_CAPTURE_LAYERS)\n'
            '        _cap = __import__("os").environ.get("SGLANG_CAPTURE_LAYERS", "all")\n'
            '        if _cap.strip().lower() == "all":\n'
            '            self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '            self._capture_final_norm = True\n'
            '        else:\n'
            '            _req = {int(x) for x in _cap.split(",") if x.strip().isdigit()}\n'
            '            _valid = set(range(self.start_layer, self.end_layer))\n'
            '            self.layers_to_capture = sorted(_req & _valid)\n'
            '            self._capture_final_norm = self.end_layer in _req\n'
            '            if not self.layers_to_capture and not self._capture_final_norm:\n'
            '                self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '                self._capture_final_norm = True\n'
            '        # PATCHED: reject pipeline parallelism (aux_hidden_states lost between PP stages)\n'
            '        if self.pp_group.world_size > 1:\n'
            '            raise RuntimeError(\n'
            '                "[PATCH] Pipeline parallelism (pp_size>1) is NOT supported for "\n'
            '                "hidden states extraction. Non-last PP ranks discard "\n'
            '                "aux_hidden_states during PPProxyTensors transfer. "\n'
            '                "Use tensor parallelism (tp_size) instead."\n'
            '            )',
        ),
        # 2. Glm4MoeModel.forward(): pre-allocated buffer replaces list append.
        #    also fixes the official `hidden_states + residual` crash at layer 0 (residual=None).
        (
            '        aux_hidden_states = []\n'
            '        for i in range(normal_start_layer, normal_end_layer):\n'
            '            with get_global_expert_distribution_recorder().with_current_layer(i):\n'
            '                if i in self.layers_to_capture:\n'
            '                    aux_hidden_states.append(hidden_states + residual)\n',
            # ↓↓↓
            '        # PATCHED: pre-allocate contiguous buffer (eliminates per-layer alloc + torch.cat)\n'
            '        _n_cap = len(self.layers_to_capture) + (1 if self._capture_final_norm else 0)\n'
            '        _aux_buf = torch.empty(\n'
            '            hidden_states.shape[0], _n_cap, hidden_states.shape[-1],\n'
            '            device=hidden_states.device, dtype=hidden_states.dtype,\n'
            '        ) if _n_cap > 0 else None\n'
            '        _aux_idx = 0\n'
            '        for i in range(normal_start_layer, normal_end_layer):\n'
            '            with get_global_expert_distribution_recorder().with_current_layer(i):\n'
            '                if i in self.layers_to_capture:\n'
            '                    # PATCHED: buffer slice write; layer-0 has residual=None\n'
            '                    _aux_buf[:, _aux_idx] = (\n'
            '                        hidden_states + residual if residual is not None else hidden_states\n'
            '                    )\n'
            '                    _aux_idx += 1\n',
        ),
        # 3. Glm4MoeModel.forward() tail: append final RMSNorm + single-tensor return
        (
            '        if len(aux_hidden_states) == 0:\n'
            '            return hidden_states\n'
            '        return hidden_states, aux_hidden_states',
            # ↓↓↓
            '        # PATCHED: append final RMSNorm + return pre-concatenated single tensor\n'
            '        if self._capture_final_norm and _aux_buf is not None:\n'
            '            _aux_buf[:, _aux_idx] = hidden_states\n'
            '            _aux_idx += 1\n'
            '        if _aux_buf is not None and _aux_idx > 0:\n'
            '            return hidden_states, [_aux_buf[:, :_aux_idx].reshape(hidden_states.shape[0], -1)]\n'
            '\n'
            '        return hidden_states',
        ),
        # 4. Glm4MoeForCausalLM: enable aux capture (unpack backbone tuple, pass through to logits_processor)
        (
            '        # For EAGLE3 support\n'
            '        self.capture_aux_hidden_states = False\n'
            '\n'
            '    def determine_num_fused_shared_experts(self):',
            # ↓↓↓
            '        # PATCHED: enable aux capture (backbone always returns tuple when patched)\n'
            '        self.capture_aux_hidden_states = True\n'
            '\n'
            '    def determine_num_fused_shared_experts(self):',
        ),
    ]


def glm4_moe_patches(version):
    """Patches for glm4_moe.py (0.5.10+ only)."""
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 model compatible with HuggingFace weights"""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 model compatible with HuggingFace weights"""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _glm4_moe_functional_patches(version)


def _glm4v_functional_patches(version):
    """Functional patches for models/glm4v.py (wrapper aux pass-through, 0.5.10+ only)."""
    return [
        # Glm4vForConditionalGeneration.forward(): unpack aux + pass to logits_processor
        (
            '        aux_hidden_states = None\n'
            '        if self.capture_aux_hidden_states:\n'
            '            hidden_states, aux_hidden_states = hidden_states\n'
            '\n'
            '        if self.pp_group.is_last_rank:\n'
            '            if not get_embedding:\n'
            '                return self.logits_processor(\n'
            '                    input_ids,\n'
            '                    hidden_states,\n'
            '                    self.lm_head,\n'
            '                    forward_batch,\n'
            '                )',
            # ↓↓↓
            '        # PATCHED: unpack aux_hidden_states from backbone (tuple = capture enabled)\n'
            '        aux_hidden_states = None\n'
            '        if isinstance(hidden_states, tuple):\n'
            '            hidden_states, aux_hidden_states = hidden_states\n'
            '\n'
            '        if self.pp_group.is_last_rank:\n'
            '            if not get_embedding:\n'
            '                return self.logits_processor(\n'
            '                    input_ids,\n'
            '                    hidden_states,\n'
            '                    self.lm_head,\n'
            '                    forward_batch,\n'
            '                    # PATCHED: pass aux through to LogitsProcessor FULL-cat path\n'
            '                    aux_hidden_states,\n'
            '                )',
        ),
    ]


def glm4v_patches(version):
    """Patches for glm4v.py (0.5.10+ only).

    Glm4vMoeForConditionalGeneration (GLM-4.5V) inherits the
    Glm4vForConditionalGeneration forward, so patching glm4v.py covers both.
    The official forward already unpacks aux but does NOT pass it through to the
    logits_processor (aux is dropped); this changes it to an isinstance unpack +
    pass-through, structurally identical to the qwen3_vl.py wrapper patch.
    """
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only GLM-4.1V model compatible with HuggingFace weights."""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only GLM-4.1V model compatible with HuggingFace weights."""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _glm4v_functional_patches(version)


def _glm4v_processor_functional_patches(version):
    """Functional patches for multimodal/processors/glm4v.py (0.5.10+ only)."""
    return [
        # 1. wrapped-block image_token + expanded-block matching regex
        (
            '        self.mm_tokens = MultimodalSpecialTokens(\n'
            '            image_token=self.IMAGE_TOKEN,\n'
            '            image_token_id=self.IM_TOKEN_ID,\n'
            '            video_token=self.VIDEO_TOKEN,\n',
            # ↓↓↓
            '        # PATCHED: wrap image token with begin/end markers and match the\n'
            '        # EXPANDED image block as ONE occurrence (same trick as qwen_vl\n'
            '        # processor); otherwise N expanded <|image|> vs 1 image ->\n'
            '        # StopIteration in legacy_load_mm_data.\n'
            '        self.mm_tokens = MultimodalSpecialTokens(\n'
            '            image_token=self.IMAGE_START_TOKEN + self.IMAGE_TOKEN + self.IMAGE_END_TOKEN,\n'
            '            image_token_id=self.IM_TOKEN_ID,\n'
            '            image_token_regex=re.compile(\n'
            '                r"<\\|begin_of_image\\|>(?:<\\|image\\|>)+<\\|end_of_image\\|>"\n'
            '            ),\n'
            '            video_token=self.VIDEO_TOKEN,\n',
        ),
        # 2. force the LEGACY PATH (same rationale as the qwen_vl processor)
        #    the block-level regex makes cnt[IMAGE] == n_image -> the FAST PATH is
        #    taken, and the FAST PATH hands the raw text containing N expanded
        #    tokens straight to the HF processor -> the 2nd occurrence raises an
        #    IndexError. The LEGACY PATH rebuilds a single wrapped block.
        (
            'class Glm4vImageProcessor(SGLangBaseProcessor):\n'
            '    models = [',
            # ↓↓↓
            'class Glm4vImageProcessor(SGLangBaseProcessor):\n'
            '    # PATCHED: force LEGACY PATH in base_processor.load_mm_data\n'
            '    # so HF processor receives a single wrapped image block per image\n'
            '    # (aligned with apply_chat_template output) -> seq_len stays consistent.\n'
            '    support_dynamic_frame_expansion = True\n'
            '    models = [',
        ),
    ]


def glm4v_processor_patches(version):
    """Patches for multimodal/processors/glm4v.py (0.5.10+ only).

    Force GLM-V's image placeholder to be the WRAPPED block form and match the
    EXPANDED image block as ONE occurrence (same trick as QwenVLImageProcessor):
      image_token: "<|image|>" -> "<|begin_of_image|><|image|><|end_of_image|>"
      image_token_regex: matches <|begin_of_image|>(<|image|>)+<|end_of_image|>
    and forces the LEGACY PATH (support_dynamic_frame_expansion=True) so the HF
    processor receives single-wrapped-block text whose re-expansion matches the
    original input_ids exactly.
    """
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker + import re (this file originally has no re import)
        (
            'from typing import List, Union',
            # ↓↓↓
            f'# {marker}\n'
            'import re\n'
            'from typing import List, Union',
        ),
    ] + _glm4v_processor_functional_patches(version)


def _deepseek_v4_functional_patches(version):
    """Patches for deepseek_v4.py (0.5.13+ only; no such file on 0.5.10).

    DeepSeek-V4 is an mHC architecture: the residual stream is hc_mult(=4)-way
    [tokens, hc_mult, hidden]. Capture semantics: at every layer entry, this
    layer's Attention hc_pre merges the mHC streams, taking the pre-RMSNorm
    [tokens, hidden]; the last layer takes the final RMSNorm output (2D) --
    consistent with PlugGuard's layer-representation convention
    (0..N-1 = each layer's entry hc_pre, pre-norm; N = final norm) with the
    feature dim unchanged. The fusion path first restores the previous layer's
    deferred hc_post; the capture bypass never modifies backbone state.
    """
    if version != "0.5.13":
        return []
    return [
        # 1. Configurable layer capture via env SGLANG_CAPTURE_LAYERS + PP reject
        (
            '        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()\n'
            '        self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()\n'
            '        if self.dsa_enable_prefill_cp:\n'
            '            self.cp_size = get_attention_cp_size()',
            # ↓↓↓
            '        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()\n'
            '        self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()\n'
            '        if self.dsa_enable_prefill_cp:\n'
            '            self.cp_size = get_attention_cp_size()\n'
            '\n'
            '        # PATCHED: configurable layer capture (env SGLANG_CAPTURE_LAYERS)\n'
            '        _cap = __import__("os").environ.get("SGLANG_CAPTURE_LAYERS", "all")\n'
            '        if _cap.strip().lower() == "all":\n'
            '            self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '            self._capture_final_norm = True\n'
            '        else:\n'
            '            _req = {int(x) for x in _cap.split(",") if x.strip().isdigit()}\n'
            '            _valid = set(range(self.start_layer, self.end_layer))\n'
            '            self.layers_to_capture = sorted(_req & _valid)\n'
            '            self._capture_final_norm = self.end_layer in _req\n'
            '            if not self.layers_to_capture and not self._capture_final_norm:\n'
            '                self.layers_to_capture = list(range(self.start_layer, self.end_layer))\n'
            '                self._capture_final_norm = True\n'
            '        # PATCHED: reject pipeline parallelism (aux capture lost between PP stages)\n'
            '        if self.pp_group.world_size > 1:\n'
            '            raise RuntimeError(\n'
            '                "[PATCH] Pipeline parallelism (pp_size>1) is NOT supported for "\n'
            '                "hidden states extraction. "\n'
            '                "Use tensor parallelism (tp_size) instead."\n'
            '            )',
        ),
        # 2. DeepseekV4Model.forward(): inject capture at loop head (pre-allocated buffer + hc_pre read before norm)
        (
            '        use_fused = self.use_fused_mhc_post_pre\n'
            '        prev_residual, prev_post, prev_comb = None, None, None\n'
            '        last_layer = None\n'
            '        for i in range(self.start_layer, self.end_layer):\n'
            '            layer = self.layers[i]\n'
            '            last_layer = layer\n',
            # ↓↓↓
            '        # PATCHED: pre-allocate capture buffer (attention hc_pre, before RMSNorm)\n'
            '        _n_cap = len(self.layers_to_capture) + (1 if self._capture_final_norm else 0)\n'
            '        _aux_buf = torch.empty(\n'
            '            hidden_states.shape[0], _n_cap, self.hidden_size,\n'
            '            device=hidden_states.device, dtype=hidden_states.dtype,\n'
            '        ) if _n_cap > 0 else None\n'
            '        _aux_idx = 0\n'
            '        use_fused = self.use_fused_mhc_post_pre\n'
            '        prev_residual, prev_post, prev_comb = None, None, None\n'
            '        last_layer = None\n'
            '        for i in range(self.start_layer, self.end_layer):\n'
            '            if _aux_buf is not None and i in self.layers_to_capture:\n'
            '                # PATCHED: capture attention hc_pre output before RMSNorm\n'
            '                # fused post-pre path: layer i-1 defers its FFN hc_post, so the\n'
            '                # loop-head hidden_states is the raw 2D FFN output; complete the\n'
            '                # deferred hc_post first (mirrors official dspark capture).\n'
            '                if use_fused and prev_residual is not None:\n'
            '                    _hs_cap = self.layers[i - 1].hc_post(\n'
            '                        hidden_states, prev_residual, prev_post, prev_comb\n'
            '                    )\n'
            '                else:\n'
            '                    _hs_cap = hidden_states\n'
            '                # Use the pretrained per-layer readout, without input_layernorm.\n'
            '                _cap_layer = self.layers[i]\n'
            '                _cap_y, _, _, _cap_norm_fused = _cap_layer.hc_pre(\n'
            '                    _hs_cap, _cap_layer.hc_attn_fn,\n'
            '                    _cap_layer.hc_attn_scale, _cap_layer.hc_attn_base,\n'
            '                    norm=None,\n'
            '                )\n'
            '                if _cap_norm_fused:\n'
            '                    raise RuntimeError("hc_pre capture must be before RMSNorm")\n'
            '                _aux_buf[:, _aux_idx] = _cap_y\n'
            '                _aux_idx += 1\n'
            '            layer = self.layers[i]\n'
            '            last_layer = layer\n',
        ),
        # 3. DeepseekV4Model.forward() tail: append final norm + wrap aux for the wrapper to unpack
        (
            '        pre_hc_head = hidden_states.flatten(1)\n'
            '\n'
            '        hidden_states = self.hc_head(\n'
            '            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base\n'
            '        )\n'
            '        hidden_states = self.norm(hidden_states)\n'
            '\n'
            '        return hidden_states, pre_hc_head',
            # ↓↓↓
            '        pre_hc_head = hidden_states.flatten(1)\n'
            '\n'
            '        hidden_states = self.hc_head(\n'
            '            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base\n'
            '        )\n'
            '        hidden_states = self.norm(hidden_states)\n'
            '\n'
            '        # PATCHED: append final RMSNorm output + wrap aux for wrapper unpack\n'
            '        if self._capture_final_norm and _aux_buf is not None:\n'
            '            _aux_buf[:, _aux_idx] = hidden_states\n'
            '            _aux_idx += 1\n'
            '        if _aux_buf is not None and _aux_idx > 0:\n'
            '            return (hidden_states, pre_hc_head), [\n'
            '                _aux_buf[:, :_aux_idx].reshape(hidden_states.shape[0], -1)\n'
            '            ]\n'
            '\n'
            '        return hidden_states, pre_hc_head',
        ),
        # 4. DeepseekV4ForCausalLM: enable aux capture (the official unpack/pass-through logic already exists)
        (
            '        self.logits_processor = LogitsProcessor(config)\n'
            '        self.capture_aux_hidden_states = False\n'
            '        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)',
            # ↓↓↓
            '        self.logits_processor = LogitsProcessor(config)\n'
            '        # PATCHED: enable aux capture (backbone returns wrapped tuple when patched)\n'
            '        self.capture_aux_hidden_states = True\n'
            '        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)',
        ),
    ]


def deepseek_v4_patches(version):
    """Patches for deepseek_v4.py (0.5.13+ only)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner (MUST be placed AFTER `from __future__ import ...`)
        (
            'from __future__ import annotations\n'
            '\n'
            'import concurrent.futures\n'
            'import logging',
            # ↓↓↓
            'from __future__ import annotations\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            '\n'
            'import concurrent.futures\n'
            'import logging',
        ),
    ] + _deepseek_v4_functional_patches(version)


def _memory_pool_0513_functional_patches(version):
    """Patch for mem_cache/memory_pool.py (0.5.13+): ROCm fp8 fnuz dtype compat.

    The 0.5.13 KVCache lowers store_dtype to uint8 only when dtype is in
    {e5m2, e4m3fn}; on ROCm fp8_e4m3 maps to float8_e4m3fnuz, which is not in
    the list -> DeepSeekV4SingleKVPool create_buffer's
    assert store_dtype == torch.uint8 fails. Upstream main already fixed this
    (the condition now includes e4m3fnuz); this patch aligns with it.
    """
    if version != "0.5.13":
        return []
    return [
        (
            '        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):',
            # ↓↓↓
            '        # PATCHED: include ROCm fnuz fp8 dtype (align with upstream main)\n'
            '        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):',
        ),
    ]


def memory_pool_0513_patches(version):
    """Patches for mem_cache/memory_pool.py (0.5.13+ only)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal mem pool code)
        (
            'KVCache actually holds the physical kv cache.\n"""\n\nfrom __future__ import annotations',
            # ↓↓↓
            'KVCache actually holds the physical kv cache.\n"""\n'
            f'# {marker}\n\n'
            'from __future__ import annotations',
        ),
    ] + _memory_pool_0513_functional_patches(version)


def _dsv4_mhc_rocm_functional_patches(version):
    """Patches for layers/mhc.py (0.5.13+, ROCm gfx942): mhc_pre splitk tile fits 64KB LDS.

    Upstream picks hidden_block=256 for hc_hidden_size==16384: x_smem(16KB) +
    fn_smem(32KB) double-buffered via num_stages=2 totals 96KB of dynamic shared
    memory, designed for NVIDIA opt-in large smem. gfx942's LDS cap is 64KB, and
    the tilelang HIP wrapper's pre-launch check rejects it outright
    ("Requested dynamic shared memory 98304 exceeds device limit 65536").
    Lowered to 128 (48KB total), matching the hc_hidden_size==28672 branch.
    Decided at runtime via torch.version.hip; the CUDA path keeps 256.
    """
    if version != "0.5.13":
        return []
    return [
        (
            '            if hc_hidden_size == 16384:\n'
            '                hidden_block = 256',
            # ↓↓↓
            '            if hc_hidden_size == 16384:\n'
            '                # PATCHED(ROCm): gfx942 LDS 64KB cap (256 -> 96KB smem overflow)\n'
            '                hidden_block = 128 if torch.version.hip else 256',
        ),
        # the non-splitk mhc_pre_gemm_sqrsum_tilelang (tokens>2048 path): same smem overflow
        (
            '    token_block: int = 32,\n'
            '    hidden_block: int = 256,\n'
            ') -> tilelang.JITKernel:',
            # ↓↓↓
            '    token_block: int = 32,\n'
            '    # PATCHED(ROCm): gfx942 LDS 64KB cap (256 -> 96KB smem overflow)\n'
            '    hidden_block: int = 128 if torch.version.hip else 256,\n'
            ') -> tilelang.JITKernel:',
        ),
    ]


def dsv4_mhc_rocm_patches(version):
    """Patches for layers/mhc.py (0.5.13+ only; ROCm gfx942 tile fix)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal kernel code)
        (
            'import functools\n'
            'import math',
            # ↓↓↓
            f'# {marker}\n'
            'import functools\n'
            'import math',
        ),
    ] + _dsv4_mhc_rocm_functional_patches(version)


def _dsv4_tilelang_kernel_rocm_functional_patches(version):
    """Patches for layers/attention/dsa/tilelang_kernel.py (0.5.13+, ROCm gfx942).

    The gfx942 branch's num_stages=1 trips the tilelang PipelineInjector
    validation ("two statements with buffer access dependency in the same stage
    of the software pipeline cannot be reordered"); the gfx950 branch already
    uses 0, so align with it. This branch only fires on hip and non-gfx950; the
    CUDA path is unaffected.
    """
    if version != "0.5.13":
        return []
    return [
        (
            '        block_I, threads, num_stages, block_per_cu, cu = 32, 128, 1, 1, 304',
            # ↓↓↓
            '        # PATCHED(ROCm): num_stages=1 trips tilelang pipeliner validation; align with gfx950 (0)\n'
            '        block_I, threads, num_stages, block_per_cu, cu = 32, 128, 0, 1, 304',
        ),
    ]


def dsv4_tilelang_kernel_rocm_patches(version):
    """Patches for layers/attention/dsa/tilelang_kernel.py (0.5.13+ only)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal kernel code)
        (
            'import functools\n'
            'from functools import lru_cache',
            # ↓↓↓
            f'# {marker}\n'
            'import functools\n'
            'from functools import lru_cache',
        ),
    ] + _dsv4_tilelang_kernel_rocm_functional_patches(version)


def _dsv4_compress_rocm_functional_patches(version):
    """Patches for jit_kernel/dsv4/compress.py (0.5.13+, ROCm): dtype alignment.

    The ROCm aiter tgemm path's compute_kv_score outputs bf16 (compressor.py),
    while the kv_score state buffer is the architecture-constant fp32
    (model_runner_kv_cache_mixin.py state_dtype); tvm_ffi JIT kernels match
    parameter dtypes strictly -> an fp32 buffer hits a bf16 instantiation error.
    On the CUDA path kv_score is fp32 already (linear_bf16_fp32 fallback), so the
    condition never fires. Align the input to the buffer dtype before creating
    out, making the whole chain match CUDA
    (the downstream norm_rope kernel instantiates fp32 following kv.dtype,
    matching the fp32 norm weights).
    """
    if version != "0.5.13":
        return []
    return [
        (
            '    if out is None:\n'
            '        num_q_tokens = plan[1].shape[0]  # NOTE: decode = bs, prefill = dynamic',
            # ↓↓↓
            '    # PATCHED(ROCm): aiter tgemm yields bf16 input but state buffer is fp32\n'
            '    # (arch constant); align BEFORE out creation so the chain stays consistent.\n'
            '    if kv_score_input.dtype != kv_score_buffer.dtype:\n'
            '        kv_score_input = kv_score_input.to(kv_score_buffer.dtype)\n'
            '    if out is None:\n'
            '        num_q_tokens = plan[1].shape[0]  # NOTE: decode = bs, prefill = dynamic',
        ),
    ]


def dsv4_compress_rocm_patches(version):
    """Patches for jit_kernel/dsv4/compress.py (0.5.13+ only; ROCm dtype fix)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal jit kernel code)
        (
            'from __future__ import annotations\n'
            '\n'
            'from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Union',
            # ↓↓↓
            'from __future__ import annotations\n'
            f'# {marker}\n'
            '\n'
            'from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Union',
        ),
    ] + _dsv4_compress_rocm_functional_patches(version)


def _dsv4_indexer_rocm_functional_patches(version):
    """Patches for layers/attention/dsv4/indexer.py (0.5.13+, ROCm): metadata field guard.

    forward_c4_indexer accesses core_metadata.c4_sparse_raw_indices directly;
    the field is defined only on the CUDA backend's metadata dataclass
    (deepseek_v4_backend.py), not on the hip_radix backend's DSV4AttnMetadata
    -> AttributeError. With the getattr guard, a missing field becomes None and
    control falls into the SGLANG_TOPK_TRANSFORM_512_TORCH torch implementation
    (which accepts raw_indices=None).
    """
    if version != "0.5.13":
        return []
    return [
        (
            '        elif core_metadata.c4_sparse_raw_indices is not None:',
            # ↓↓↓
            '        # PATCHED(ROCm): hip_radix metadata lacks this field (CUDA-only)\n'
            '        elif getattr(core_metadata, "c4_sparse_raw_indices", None) is not None:',
        ),
    ]


def dsv4_indexer_rocm_patches(version):
    """Patches for layers/attention/dsv4/indexer.py (0.5.13+ only)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal attention code)
        (
            'from __future__ import annotations\n'
            '\n'
            'from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypeAlias, Union',
            # ↓↓↓
            'from __future__ import annotations\n'
            f'# {marker}\n'
            '\n'
            'from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypeAlias, Union',
        ),
    ] + _dsv4_indexer_rocm_functional_patches(version)


def _logits_processor_0513_functional_patches(version):
    """Patches for layers/logits_processor.py (0.5.13+): aux capture wins over before_norm.

    Official logic: when hidden_states_before_norm (dsv4's pre_hc_head) exists it
    unconditionally overwrites the hidden states to store -- our all-layer aux
    capture buffer would be lost. Changed so before_norm only applies when no aux
    exists: with aux present (this patch's captured activations) aux is stored;
    the official no-aux scenarios (EAGLE etc.) behave unchanged. For other models
    before_norm is None and nothing changes.
    """
    if version != "0.5.13":
        return []
    return [
        (
            '        if hidden_states_to_store_before_norm is not None:\n'
            '            # NOTE: when hidden_states_before_norm is provided, we always\n'
            '            # prefer to return it.\n'
            '            hidden_states_to_store = hidden_states_to_store_before_norm',
            # ↓↓↓
            '        # PATCHED: keep aux capture when present (aux != None means the dsv4\n'
            '        # all-layer capture is active); before_norm wins only without aux.\n'
            '        if hidden_states_to_store_before_norm is not None and aux_hidden_states is None:\n'
            '            hidden_states_to_store = hidden_states_to_store_before_norm',
        ),
    ]


def logits_processor_0513_patches(version):
    """Patches for layers/logits_processor.py (0.5.13+ only; aux-over-before_norm)."""
    if version != "0.5.13":
        return []
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time marker (no banner print — internal logits code)
        (
            '\"\"\"Logits processing.\"\"\"\n'
            '\n'
            'import dataclasses',
            # ↓↓↓
            '\"\"\"Logits processing.\"\"\"\n'
            f'# {marker}\n'
            '\n'
            'import dataclasses',
        ),
    ] + _logits_processor_0513_functional_patches(version)


def _scheduler_functional_patches(version):
    """Patches for scheduler_output_processor_mixin.py.

    Replace .cpu().clone().tolist() with .cpu().float() to keep data as
    torch.Tensor instead of nested Python lists.  This eliminates the
    massive overhead of creating millions of Python float objects and
    greatly speeds up pickle IPC serialization.
    """
    return [
        # 1. Prefill path: hidden_state_offset slicing
        (
            '                            ]\n'
            '                            .cpu()\n'
            '                            .clone()\n'
            '                            .tolist()\n'
            '                        )',
            # ↓↓↓
            '                            ]\n'
            '                            # PATCHED: keep as tensor (was .cpu().clone().tolist())\n'
            '                            .cpu()\n'
            '                            .float()\n'
            '                        )',
        ),
        # 2. Decode path: per-request index
        (
            'logits_output.hidden_states[i].cpu().clone().tolist()',
            # ↓↓↓
            '# PATCHED: keep as tensor (was .cpu().clone().tolist())\n'
            '                    logits_output.hidden_states[i].cpu().float()',
        ),
    ]


def _qwen3_moe_functional_patches(version):
    """Patch for qwen3_moe.py (Qwen3MoeForCausalLM)."""
    return [
        # 2. Always enable capture_aux_hidden_states in Qwen3MoeForCausalLM
        (
            '        self.logits_processor = LogitsProcessor(config)\n'
            '        self.capture_aux_hidden_states = False',
            # ↓↓↓
            '        self.logits_processor = LogitsProcessor(config)\n'
            '        # PATCHED: always capture aux hidden states (original: False)\n'
            '        self.capture_aux_hidden_states = True',
        ),
    ]


def _layernorm_functional_patches(version):
    """Patch for layernorm.py: fix forward_hip incompatibility with vllm ROCm build.

    The vllm ROCm build's fused_add_rms_norm may be incompatible with the
    6-argument API sglang expects. This patch forces the old 4-argument
    in-place path.
    """
    # 0.5.10: forward_hip takes 6 arguments directly, no version branch
    # NOTE: at the top of forward_hip in the sglang 0.5.10.post1 source there is
    # `w = self.weight.data + 1.0`; every later call uses the local w instead of
    # self.weight.data, so the patch must match w.
    # Also, the vllm branch sits at nesting level 3 inside the `else:` block (12
    # spaces), preceded by `if not x.is_contiguous(): x = x.contiguous()`.
    return [
        (
            '            if residual is not None:\n'
            '                out = torch.empty_like(x)\n'
            '                residual_out = torch.empty_like(x)\n'
            '                if post_residual_addition is not None:\n'
            '                    residual = residual + post_residual_addition\n'
            '                fused_add_rms_norm(\n'
            '                    out, x, residual_out, residual, w, self.variance_epsilon\n'
            '                )\n'
            '                return out, residual_out',
            # ↓↓↓ PATCHED: force 4-arg in-place API (compatible with vllm ROCm builds)
            '            if residual is not None:\n'
            '                # PATCHED: force 4-arg in-place fused_add_rms_norm (vllm ROCm compat)\n'
            '                if post_residual_addition is not None:\n'
            '                    residual = residual + post_residual_addition\n'
            '                fused_add_rms_norm(x, residual, w, self.variance_epsilon)\n'
            '                return x, residual',
        ),
        # RMSNorm.forward_hip (8-space indent, self.weight.data variant): the same
        # 6->4 argument fix. The GLM-4.5V vision tower / text backbone uses this
        # class; it only fires in environments with vllm installed (notebooks etc.)
        # (cluster task environments have no vllm -> _has_vllm_rms_norm=False ->
        # forward_native, unaffected).
        (
            '        if residual is not None:\n'
            '            out = torch.empty_like(x)\n'
            '            residual_out = torch.empty_like(x)\n'
            '            if post_residual_addition is not None:\n'
            '                residual = residual + post_residual_addition\n'
            '            fused_add_rms_norm(\n'
            '                out, x, residual_out, residual, self.weight.data, self.variance_epsilon\n'
            '            )\n'
            '            return out, residual_out',
            # ↓↓↓ PATCHED: force 4-arg in-place API (compatible with vllm ROCm builds)
            '        if residual is not None:\n'
            '            # PATCHED: force 4-arg in-place fused_add_rms_norm (vllm ROCm compat)\n'
            '            if post_residual_addition is not None:\n'
            '                residual = residual + post_residual_addition\n'
            '            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)\n'
            '            return x, residual',
        ),
    ]


def _unquant_functional_patches(version):
    """Patch for unquant.py: per-expert shuffle to reduce peak memory.

    The original implementation shuffles the whole [num_experts, N, K] at once,
    peaking at 2x the full weights. Changed to a per-expert loop that shuffles
    and writes back in place, cutting the peak to full + 2/num_experts.
    For a 128-expert MoE model that is 2x -> ~1.016x.
    """
    # 0.5.10: uses _should_use_aiter_moe + copy_or_rebind_param
    return [
        (
            '    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
            '        # Skip aiter weight shuffle when using non-auto MoE backend (e.g., triton, triton_kernels)\n'
            '        # because aiter CK kernels don\'t support all GEMM dimensions\n'
            '        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()\n'
            '        if _should_use_aiter_moe:\n'
            '            copy_or_rebind_param(\n'
            '                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))\n'
            '            )\n'
            '            torch.cuda.empty_cache()\n'
            '            copy_or_rebind_param(\n'
            '                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))\n'
            '            )\n'
            '            torch.cuda.empty_cache()',
            # ↓↓↓ PATCHED: per-expert shuffle to reduce peak memory (2× → ~1×)
            '    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n'
            '        # Skip aiter weight shuffle when using non-auto MoE backend (e.g., triton, triton_kernels)\n'
            '        # because aiter CK kernels don\'t support all GEMM dimensions\n'
            '        _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()\n'
            '        if _should_use_aiter_moe:\n'
            '            # PATCHED: per-expert shuffle to reduce peak memory\n'
            '            _ne = layer.w13_weight.shape[0]\n'
            '            for _i in range(_ne):\n'
            '                layer.w13_weight.data[_i] = shuffle_weight(\n'
            '                    layer.w13_weight.data[_i].unsqueeze(0).contiguous(), (16, 16)\n'
            '                ).squeeze(0)\n'
            '            torch.cuda.empty_cache()\n'
            '            for _i in range(_ne):\n'
            '                layer.w2_weight.data[_i] = shuffle_weight(\n'
            '                    layer.w2_weight.data[_i].unsqueeze(0).contiguous(), (16, 16)\n'
            '                ).squeeze(0)\n'
            '            torch.cuda.empty_cache()',
        ),
    ]


def _forward_mha_functional_patches(version):
    """Patch for deepseek_common/attention_forward_methods/forward_mha.py.

    sglang 0.5.10's _concat_and_cast_mha_k unconditionally calls
    concat_and_cast_mha_k_triton on the ROCm+aiter branch, but that triton kernel
    internally uses tl.arange(0, nope_dim), which requires nope_dim to be a power
    of 2. GLM-MoE-DSA (GLM5.1) with qk_nope_head_dim=192 triggers the
    `arange's range must be a power of 2` error.

    This patch guards the branch: the triton path is taken only when
    qk_nope_head_dim, qk_rope_head_dim and num_local_heads are all powers of 2;
    otherwise it falls through to the plain PyTorch fallback below.

    Impact on other models:
      NVIDIA + any model            never enters the _is_hip branch, unaffected
      ROCm + all dims powers of 2   guard passes, still triton, unaffected
                                    (DeepSeek-V3/R1/Kimi K2/GLM4Moe/Mistral Large 3 etc.)
      ROCm + any non-pow2 dim       guard fails, falls back to the PyTorch path:
                                    crashes become runnable
                                    (GLM-MoE-DSA / GLM5.1 / similar future models)
    """
    return [
        (
            '        elif _is_hip and self.current_attention_backend == "aiter":\n'
            '            k = k_nope.new_empty(*k_shape)\n'
            '            concat_and_cast_mha_k_triton(k, k_nope, k_pe)',
            # ↓↓↓
            '        elif (\n'
            '            _is_hip\n'
            '            and self.current_attention_backend == "aiter"\n'
            '            # PATCHED: guard triton kernel that assumes head_dim is power of 2\n'
            '            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim\n'
            '            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim\n'
            '            and next_power_of_2(self.num_local_heads) == self.num_local_heads\n'
            '        ):\n'
            '            k = k_nope.new_empty(*k_shape)\n'
            '            concat_and_cast_mha_k_triton(k, k_nope, k_pe)',
        ),
    ]


def _compute_patch_digest(version):
    """Compute a short SHA-256 digest of all functional patch content.

    The digest changes whenever any find/replace text is modified, added,
    or removed — making it possible to detect stale (outdated) patches.
    """
    all_patches = (
        _qwen3_functional_patches(version)
        + _qwen3_moe_functional_patches(version)
        + _qwen3_5_functional_patches(version)
        + _qwen3_vl_functional_patches(version)
        + _qwen_vl_processor_functional_patches(version)
        + _glm4_moe_functional_patches(version)
        + _glm4v_functional_patches(version)
        + _glm4v_processor_functional_patches(version)
        + _deepseek_v4_functional_patches(version)
        + _scheduler_0513_functional_patches(version)
        + _memory_pool_0513_functional_patches(version)
        + _dsv4_mhc_rocm_functional_patches(version)
        + _dsv4_tilelang_kernel_rocm_functional_patches(version)
        + _dsv4_compress_rocm_functional_patches(version)
        + _dsv4_indexer_rocm_functional_patches(version)
        + _logits_processor_0513_functional_patches(version)
        + _forward_mha_functional_patches(version)
        + _scheduler_functional_patches(version)
        + _layernorm_functional_patches(version)
        + _unquant_functional_patches(version)
    )
    raw = "\n".join(f"{f}|||{r}" for f, r in all_patches)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _get_patch_marker(version):
    """Return the patch marker string for a specific sglang version."""
    digest = _compute_patch_digest(version)
    return f"{_PATCH_MARKER_BASE}-{digest}"


def qwen3_patches(version):
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '# Adapted from qwen2.py\n'
            'import logging',
            # ↓↓↓
            '# Adapted from qwen2.py\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _qwen3_functional_patches(version)


def _scheduler_0513_functional_patches(version):
    """Patches for scheduler_components/batch_result_processor.py (0.5.13+).

    Since 0.5.13 the IPC serialization of hidden_states moved from
    scheduler_output_processor_mixin.py to batch_result_processor.py. Same
    .cpu().clone().tolist() -> .cpu().float() change (keep tensors; avoids the
    overhead of tens of millions of Python float objects).
    """
    if version != "0.5.13":
        return []
    return [
        # 1. Prefill path: hidden_state_offset slicing
        (
            '            ]\n'
            '            .cpu()\n'
            '            .clone()\n'
            '            .tolist()\n'
            '        )',
            # ↓↓↓
            '            ]\n'
            '            # PATCHED: keep as tensor (was .cpu().clone().tolist())\n'
            '            .cpu()\n'
            '            .float()\n'
            '        )',
        ),
        # 2. Decode path A (spec_v1 branch, 16-space indent)
        (
            '                if req.return_hidden_states and logits_output.hidden_states is not None:\n'
            '                    req.hidden_states.append(\n'
            '                        logits_output.hidden_states[i].cpu().clone().tolist()\n'
            '                    )\n'
            '                if req.grammar is not None:\n'
            '                    req.grammar.finished = req.finished()',
            # ↓↓↓
            '                if req.return_hidden_states and logits_output.hidden_states is not None:\n'
            '                    req.hidden_states.append(\n'
            '                        # PATCHED: keep as tensor (was .cpu().clone().tolist())\n'
            '                        logits_output.hidden_states[i].cpu().float()\n'
            '                    )\n'
            '                if req.grammar is not None:\n'
            '                    req.grammar.finished = req.finished()',
        ),
        # 3. Decode path B (regular branch, 12-space indent)
        (
            '            if req.return_hidden_states and logits_output.hidden_states is not None:\n'
            '                req.hidden_states.append(\n'
            '                    logits_output.hidden_states[i].cpu().clone().tolist()\n'
            '                )\n'
            '\n'
            '            if req.grammar is not None:',
            # ↓↓↓
            '            if req.return_hidden_states and logits_output.hidden_states is not None:\n'
            '                req.hidden_states.append(\n'
            '                    # PATCHED: keep as tensor (was .cpu().clone().tolist())\n'
            '                    logits_output.hidden_states[i].cpu().float()\n'
            '                )\n'
            '\n'
            '            if req.grammar is not None:',
        ),
    ]


def scheduler_patches(version):
    marker = _get_patch_marker(version)
    return [
        # 1. Marker (no banner print — internal scheduler code)
        (
            'from __future__ import annotations\n\nimport logging',
            # ↓↓↓
            'from __future__ import annotations\n'
            f'# {marker}\n\n'
            'import logging',
        ),
    ] + (
        _scheduler_0513_functional_patches(version)
        if version == "0.5.13"
        else _scheduler_functional_patches(version)
    )


def qwen3_moe_patches(version):
    marker = _get_patch_marker(version)
    return [
        # 1. Load-time banner
        (
            '"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""\n'
            '\n'
            'import logging',
            # ↓↓↓
            '"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""\n'
            f'# {marker}\n'
            f'{_BANNER_LINE}\n'
            'import logging',
        ),
    ] + _qwen3_moe_functional_patches(version)


# =====================================================================
# Core logic
# =====================================================================

def _apply_patches(filepath, patches):
    """Apply a list of (find, replace) patches to a file. Errors on ambiguity."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    for i, (find_text, replace_text) in enumerate(patches, 1):
        count = content.count(find_text)
        if count == 0:
            raise RuntimeError(
                f"Patch #{i} target NOT FOUND in {filepath}:\n"
                f"  >>> {find_text[:100]}..."
            )
        if count > 1:
            raise RuntimeError(
                f"Patch #{i} target found {count} times in {filepath} (expected 1):\n"
                f"  >>> {find_text[:100]}..."
            )
        content = content.replace(find_text, replace_text)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def _is_patched(filepath, version):
    """Check if a file contains our CURRENT patch marker (incl. digest).

    Returns True only when the marker matches the current digest.
    A file patched with an older version returns False.
    """
    try:
        marker = _get_patch_marker(version)
        with open(filepath, "r", encoding="utf-8") as f:
            return marker in f.read()
    except OSError:
        return False


def _is_stale_patched(filepath, version):
    """Check if a file carries an OUTDATED version of our patch."""
    try:
        marker = _get_patch_marker(version)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return _PATCH_MARKER_BASE in content and marker not in content
    except OSError:
        return False


def layernorm_patches(version):
    """Patches for layernorm.py (ROCm vllm fused_add_rms_norm compatibility)."""
    marker = _get_patch_marker(version)
    marker_find = 'from sglang.srt.layers.utils import MultiPlatformOp'
    return [
        # Marker so _is_patched() can detect it
        (
            marker_find,
            # ↓↓↓
            f'# {marker}\n'
            + marker_find,
        ),
    ] + _layernorm_functional_patches(version)


def unquant_patches(version):
    """Patches for unquant.py (per-expert shuffle to reduce peak memory)."""
    marker = _get_patch_marker(version)
    return [
        # Marker so _is_patched() can detect it
        (
            '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip',
            # ↓↓↓
            f'# {marker}\n'
            '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip',
        ),
    ] + _unquant_functional_patches(version)


def forward_mha_patches(version):
    """Patches for deepseek_common/attention_forward_methods/forward_mha.py
    (0.5.10+ only; ROCm head_dim power-of-2 guard).
    """
    marker = _get_patch_marker(version)
    return [
        # Marker so _is_patched() can detect it (next_power_of_2 is already imported
        # at module top in 0.5.10, so the guard expression compiles without changes).
        (
            'from sglang.srt.utils import BumpAllocator, get_bool_env_var, next_power_of_2',
            # ↓↓↓
            f'# {marker}\n'
            'from sglang.srt.utils import BumpAllocator, get_bool_env_var, next_power_of_2',
        ),
    ] + _forward_mha_functional_patches(version)


# =====================================================================
# Path resolution & cache cleanup helpers (shared by install/uninstall/status)
# =====================================================================

def _resolve_paths(models_dir, version=None):
    """Resolve all patchable file paths, patch steps, and summaries.

    Returns a dict with:
        File paths:   qwen2, qwen3, qwen2_moe, qwen3_moe, qwen3_5, qwen3_vl,
                      qwen3_vl_moe, sched_mixin, layernorm, unquant
        Directories:  models_dir, managers_dir, layers_dir, quantization_dir
        Derived:
            all_files   — every patchable file (for uninstall / status)
            active_files — files whose patch is enabled (for install)
            cache_dirs  — __pycache__ directories to clean
            patch_steps — list of dicts describing each patch step
    """
    managers_dir = os.path.join(os.path.dirname(models_dir), "managers")
    layers_dir = os.path.join(os.path.dirname(models_dir), "layers")
    mem_cache_dir = os.path.join(os.path.dirname(models_dir), "mem_cache")
    quantization_dir = os.path.join(layers_dir, "quantization")
    multimodal_processors_dir = os.path.join(
        os.path.dirname(models_dir), "multimodal", "processors"
    )
    deepseek_common_attn_dir = os.path.join(
        models_dir, "deepseek_common", "attention_forward_methods"
    )
    dsa_attn_dir = os.path.join(layers_dir, "attention", "dsa")
    dsv4_attn_dir = os.path.join(layers_dir, "attention", "dsv4")
    jit_kernel_dsv4_dir = os.path.join(
        os.path.dirname(os.path.dirname(models_dir)), "jit_kernel", "dsv4"
    )

    _is_rocm = os.environ.get('_SGLANG_VLLM_ROCM_COMPAT') == '1'
    _is_aiter_lowmem = os.environ.get('_SGLANG_AITER_LOWMEM') == '1'
    _is_0510 = version == "0.5.10"  # qwen3_5 / qwen3_vl / glm4v_moe anchors fit 0.5.10 only
    _is_0513 = version == "0.5.13"  # deepseek_v4 fits 0.5.13 only

    paths = {
        "models_dir": models_dir,
        "managers_dir": managers_dir,
        "layers_dir": layers_dir,
        "quantization_dir": quantization_dir,
        "multimodal_processors_dir": multimodal_processors_dir,
        "deepseek_common_attn_dir": deepseek_common_attn_dir,
        "qwen3": os.path.join(models_dir, "qwen3.py"),
        "qwen3_moe": os.path.join(models_dir, "qwen3_moe.py"),
        "qwen3_5": os.path.join(models_dir, "qwen3_5.py"),
        "qwen3_vl": os.path.join(models_dir, "qwen3_vl.py"),
        "qwen3_vl_moe": os.path.join(models_dir, "qwen3_vl_moe.py"),
        "qwen_vl_processor": os.path.join(multimodal_processors_dir, "qwen_vl.py"),
        "glm4v_processor": os.path.join(multimodal_processors_dir, "glm4v.py"),
        "glm4_moe": os.path.join(models_dir, "glm4_moe.py"),
        "glm4v": os.path.join(models_dir, "glm4v.py"),
        "forward_mha": os.path.join(deepseek_common_attn_dir, "forward_mha.py"),
        "sched_mixin": (
            os.path.join(managers_dir, "scheduler_components", "batch_result_processor.py")
            if _is_0513
            else os.path.join(managers_dir, "scheduler_output_processor_mixin.py")
        ),
        "deepseek_v4": os.path.join(models_dir, "deepseek_v4.py"),
        "memory_pool": os.path.join(mem_cache_dir, "memory_pool.py"),
        "dsv4_mhc": os.path.join(layers_dir, "mhc.py"),
        "dsv4_tilelang_kernel": os.path.join(dsa_attn_dir, "tilelang_kernel.py"),
        "dsv4_compress": os.path.join(jit_kernel_dsv4_dir, "compress.py"),
        "dsv4_indexer": os.path.join(dsv4_attn_dir, "indexer.py"),
        "logits_processor": os.path.join(layers_dir, "logits_processor.py"),
        "layernorm": os.path.join(layers_dir, "layernorm.py"),
        "unquant": os.path.join(quantization_dir, "unquant.py"),
    }

    # ── Patch registry: each step is {key, fn, display, enabled, summary} ──
    paths["patch_steps"] = [
        {"key": "qwen3",      "fn": qwen3_patches,     "display": "qwen3.py",
         "enabled": not _is_0513,
         "summary": [
             "capture_aux_hidden_states = True (was: False)",
             "Load-time banner printed to stderr",
         ]},
        {"key": "qwen3_moe",  "fn": qwen3_moe_patches, "display": "qwen3_moe.py",
         "enabled": not _is_0513,
         "summary": [
             "capture_aux_hidden_states = True (was: False)",
             "Load-time banner printed to stderr",
         ]},
        {"key": "sched_mixin", "fn": scheduler_patches,
         "display": "scheduler_output_processor_mixin.py",
         "enabled": True,
         "summary": [
             ".cpu().clone().tolist() -> .cpu().float() (keep tensor)",
         ]},
        {"key": "qwen3_5",   "fn": qwen3_5_patches,
         "display": "qwen3_5.py (Qwen3.5 dense + MoE)",
         "enabled": _is_0510,
         "summary": [
             "layers_to_capture = ALL layers (was: N/A)",
             "Pre-allocated buffer for aux_hidden_states capture",
             "forward(): appends final RMSNorm output to buffer",
             "Load-time banner printed to stderr",
         ]},
        {"key": "qwen3_vl",  "fn": qwen3_vl_patches,
         "display": "qwen3_vl.py (Qwen3-VL backbone + VL wrapper)",
         "enabled": _is_0510,
         "summary": [
             "Qwen3LLMModel: pre-allocated buffer for aux_hidden_states capture",
             "Qwen3LLMModel: forward() appends final RMSNorm output to buffer",
             "VL wrapper: unpack aux_hidden_states from backbone return",
             "VL wrapper: pass aux_hidden_states to logits_processor",
             "Load-time banner printed to stderr",
         ]},
        {"key": "qwen3_vl_moe",  "fn": qwen3_vl_moe_patches,
         "display": "qwen3_vl_moe.py (Qwen3-VL MoE backbone)",
         "enabled": _is_0510,
         "summary": [
             "Qwen3MoeLLMModel: pre-allocated buffer for aux_hidden_states capture",
             "Qwen3MoeLLMModel: forward() appends final RMSNorm output to buffer",
             "(VL wrapper patch inherited from qwen3_vl.py via Qwen3VLForConditionalGeneration)",
             "Load-time banner printed to stderr",
         ]},
        {"key": "qwen_vl_processor", "fn": qwen_vl_processor_patches,
         "display": "multimodal/processors/qwen_vl.py (Qwen-VL MM processor)",
         "enabled": _is_0510,
         "summary": [
             "QwenVLImageProcessor.support_dynamic_frame_expansion = True",
             "(force LEGACY PATH → single <|image_pad|> per image → seq_len aligned)",
         ]},
        {"key": "glm4v_processor", "fn": glm4v_processor_patches,
         "display": "multimodal/processors/glm4v.py (GLM-V MM processor)",
         "enabled": _is_0510,
         "summary": [
             "image_token wrapped: <|begin_of_image|><|image|><|end_of_image|>",
             "image_token_regex matches EXPANDED block as ONE occurrence",
             "(fix StopIteration in legacy_load_mm_data for pre-expanded input_ids)",
         ]},
        {"key": "glm4_moe", "fn": glm4_moe_patches,
         "display": "glm4_moe.py (GLM-4.5/4.6/4.7 MoE + GLM-4.5V text backbone)",
         "enabled": _is_0510,
         "summary": [
             "Glm4MoeModel: layers_to_capture = ALL layers (was: empty)",
             "Glm4MoeModel: pre-allocated buffer; layer-0 residual=None guard",
             "Glm4MoeModel: forward() appends final RMSNorm output to buffer",
             "Glm4MoeForCausalLM: capture_aux_hidden_states = True (was: False)",
             "Load-time banner printed to stderr",
         ]},
        {"key": "glm4v", "fn": glm4v_patches,
         "display": "glm4v.py (GLM-4.1V/4.5V VL wrapper)",
         "enabled": _is_0510,
         "summary": [
             "VL wrapper: unpack aux_hidden_states from backbone return (isinstance tuple)",
             "VL wrapper: pass aux_hidden_states to logits_processor",
             "Load-time banner printed to stderr",
         ]},
        {"key": "deepseek_v4", "fn": deepseek_v4_patches,
         "display": "deepseek_v4.py (DeepSeek-V4, hc_pre pre-norm capture)",
         "enabled": _is_0513,
         "summary": [
             "DeepseekV4Model: layers_to_capture from env (was: N/A)",
             "DeepseekV4Model: per-layer attention hc_pre capture before RMSNorm",
             "DeepseekV4Model: forward() appends final RMSNorm + wraps aux for wrapper",
             "DeepseekV4ForCausalLM: capture_aux_hidden_states = True (was: False)",
             "Load-time banner printed to stderr",
         ]},
        {"key": "memory_pool", "fn": memory_pool_0513_patches,
         "display": "mem_cache/memory_pool.py (ROCm fp8 fnuz compat)",
         "enabled": _is_0513,
         "summary": [
             "KVCache store_dtype: include float8_e4m3fnuz -> uint8 (ROCm fp8)",
         ]},
        {"key": "dsv4_mhc", "fn": dsv4_mhc_rocm_patches,
         "display": "layers/mhc.py (ROCm gfx942 mhc_pre tile fix)",
         "enabled": _is_0513,
         "summary": [
             "mhc_pre splitk: hidden_block 256->128 on ROCm (gfx942 64KB LDS cap)",
         ]},
        {"key": "dsv4_tilelang_kernel", "fn": dsv4_tilelang_kernel_rocm_patches,
         "display": "layers/attention/dsa/tilelang_kernel.py (ROCm gfx942 pipeliner fix)",
         "enabled": _is_0513,
         "summary": [
             "dpsk_v4_fp8_partial_kernel gfx942: num_stages 1->0 (align with gfx950)",
         ]},
        {"key": "dsv4_compress", "fn": dsv4_compress_rocm_patches,
         "display": "jit_kernel/dsv4/compress.py (ROCm dtype align)",
         "enabled": _is_0513,
         "summary": [
             "compress_forward: cast kv_score_input to fp32 state-buffer dtype",
         ]},
        {"key": "dsv4_indexer", "fn": dsv4_indexer_rocm_patches,
         "display": "layers/attention/dsv4/indexer.py (ROCm metadata getattr guard)",
         "enabled": _is_0513,
         "summary": [
             "c4_sparse_raw_indices: getattr guard (hip_radix metadata lacks field)",
         ]},
        {"key": "logits_processor", "fn": logits_processor_0513_patches,
         "display": "layers/logits_processor.py (aux capture over before_norm)",
         "enabled": _is_0513,
         "summary": [
             "_get_hidden_states_to_store: aux buffer wins; before_norm as fallback",
         ]},
        {"key": "forward_mha", "fn": forward_mha_patches,
         "display": "deepseek_common/attention_forward_methods/forward_mha.py (ROCm head_dim power-of-2 guard)",
         # enabled on all platforms by default: on NVIDIA the _is_hip branch is
         # never hit (no side effects); on ROCm it is transparent for models whose
         # head dims are all powers of 2, and only bypasses the triton crash for
         # non-pow2 models (e.g. GLM5.1).
         "enabled": _is_0510,
         "summary": [
             "_concat_and_cast_mha_k: guard ROCm+aiter triton kernel with power-of-2 checks",
             "(non-pow2 head_dim automatically falls back to the PyTorch path, avoiding the tl.arange crash)",
         ]},
        {"key": "layernorm",  "fn": layernorm_patches,
         "display": "layernorm.py (ROCm compat)",
         "enabled": _is_rocm and not _is_0513,
         "summary": [
             "forward_hip(): use vllm rms_norm with ROCm compat",
         ]},
        {"key": "unquant",    "fn": unquant_patches,
         "display": "unquant.py (aiter per-expert shuffle)",
         "enabled": _is_aiter_lowmem and not _is_0513,
         "summary": [
             "process_weights_after_loading: per-expert shuffle (peak 2x -> ~1x)",
         ]},
    ]

    paths["all_files"] = [paths[s["key"]] for s in paths["patch_steps"]]
    paths["active_files"] = [paths[s["key"]] for s in paths["patch_steps"] if s["enabled"]]
    paths["cache_dirs"] = [
        os.path.join(d, "__pycache__")
        for d in (models_dir, managers_dir, layers_dir, quantization_dir,
                  multimodal_processors_dir, deepseek_common_attn_dir)
    ]
    if _is_0513:
        paths["cache_dirs"].append(
            os.path.join(managers_dir, "scheduler_components", "__pycache__")
        )
        paths["cache_dirs"].append(
            os.path.join(mem_cache_dir, "__pycache__")
        )
        paths["cache_dirs"].append(os.path.join(dsa_attn_dir, "__pycache__"))
        paths["cache_dirs"].append(os.path.join(dsv4_attn_dir, "__pycache__"))
        paths["cache_dirs"].append(os.path.join(jit_kernel_dsv4_dir, "__pycache__"))
    return paths


# Basename prefixes used to identify patch-related .pyc files
_PYC_PREFIXES = ("qwen2", "qwen3", "scheduler_output_processor_mixin.",
                 "layernorm", "unquant", "qwen3_5", "qwen3_vl",
                 "qwen_vl", "deepseek_v2", "forward_mha", "glm4_moe", "glm4v",
                 "deepseek_v4", "batch_result_processor", "memory_pool",
                 "mhc", "tilelang_kernel", "compress", "indexer",
                 "logits_processor")


def _clean_pycache(cache_dirs, verbose=True):
    """Remove patch-related .pyc files from __pycache__ directories."""
    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        try:
            entries = os.listdir(cache_dir)
        except FileNotFoundError:
            continue
        for fname in entries:
            if not any(fname.startswith(p) for p in _PYC_PREFIXES):
                continue
            fpath = os.path.join(cache_dir, fname)
            try:
                os.remove(fpath)
                if verbose:
                    print(f"[PATCH] Removed cached bytecode: {fname}")
            except FileNotFoundError:
                # multi-process concurrent-uninstall race: another rank already cleaned it, equivalent to success
                pass


def install(models_dir, version=None):
    if version is None:
        version = _detected_version or _check_version()
    patch_digest = _compute_patch_digest(version)
    p = _resolve_paths(models_dir, version=version)
    all_files = p["active_files"]

    for fpath in all_files:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Source file not found: {fpath}")

    # ── Skip if already patched with current digest ──
    if all(_is_patched(f, version) for f in all_files):
        print(f"[PATCH] Already up-to-date (v={version}, digest={patch_digest}), skipping.",
              file=sys.stderr, flush=True)
        return

    # ── Detect stale patches ──
    stale = [os.path.basename(f) for f in all_files if _is_stale_patched(f, version)]
    if stale:
        print(f"[PATCH] Stale patch detected in {', '.join(stale)} "
              f"\u2014 auto-updating to v={version}, digest={patch_digest}",
              file=sys.stderr, flush=True)

    print(f"[PATCH] sglang models dir: {models_dir}")

    # ── Backup (or restore-then-re-patch for idempotency) ──
    for fpath in all_files:
        bak = fpath + BACKUP_SUFFIX
        if os.path.isfile(bak):
            # Restore clean state from backup, then re-patch
            shutil.copy2(bak, fpath)
            print(f"[PATCH] Restored clean state from backup: {os.path.basename(fpath)}")
        else:
            shutil.copy2(fpath, bak)
            print(f"[PATCH] Backed up: {os.path.basename(fpath)} -> {os.path.basename(bak)}")

    # ── Apply patches ──
    for step in p["patch_steps"]:
        display = step["display"]
        print(f"[PATCH] Patching {display} ...")
        if not step["enabled"]:
            print(f"[PATCH] Skipped {display}")
            continue
        _apply_patches(p[step["key"]], step["fn"](version))
        print(f"[PATCH] \u2713 {display} done")

    # ── Summary ──
    print()
    print("=" * 60)
    print("[PATCH] ALL PATCHES APPLIED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("Changes made:")
    for step in p["patch_steps"]:
        if step["enabled"]:
            print(f"  {step['display']}:")
            for line in step["summary"]:
                print(f"    - {line}")
    print()
    print("To restore originals:")
    print("  python patch_sglang_all_layers.py --uninstall")


def uninstall(models_dir):
    p = _resolve_paths(models_dir)

    print(f"[PATCH] sglang models dir: {models_dir}")

    restored = 0
    for fpath in p["all_files"]:
        bak = fpath + BACKUP_SUFFIX
        if os.path.isfile(bak):
            shutil.copy2(bak, fpath)
            os.remove(bak)
            print(f"[PATCH] Restored: {os.path.basename(fpath)}")
            restored += 1
        else:
            print(f"[PATCH] No backup found for: {os.path.basename(fpath)}")

    _clean_pycache(p["cache_dirs"])

    if restored:
        print(f"\n[PATCH] ✓ Restored {restored} file(s) to original state.")
    else:
        print("\n[PATCH] Nothing to restore — no backup files found.")
    
    # clear the patched flag so a later auto_patch() in this process can run install again
    os.environ.pop(_ENV_PATCHED, None)


def auto_uninstall():
    """Auto-uninstall: find sglang models dir and restore all patched files."""
    models_dir = find_sglang_models_dir()
    uninstall(models_dir)


def status(models_dir, version=None):
    if version is None:
        version = _detected_version or _check_version()
    patch_digest = _compute_patch_digest(version)
    p = _resolve_paths(models_dir, version=version)

    print(f"[PATCH] sglang models dir: {models_dir}")
    print()
    print(f"  sglang version:       {version}")
    print(f"  Current patch digest: {patch_digest}")
    print()
    for fpath in p["all_files"]:
        name = os.path.basename(fpath)
        patched = _is_patched(fpath, version)
        stale = _is_stale_patched(fpath, version)
        bak_exists = os.path.isfile(fpath + BACKUP_SUFFIX)
        if patched:
            state = "YES (current)"
        elif stale:
            state = "STALE (outdated patch \u2014 run --install to update)"
        else:
            state = "NO"
        print(f"  {name}:")
        print(f"    patched:        {state}")
        print(f"    backup exists:  {'YES' if bak_exists else 'NO'}")


# =====================================================================
# auto_patch():  one-call entry point for extract scripts
# =====================================================================

def _auto_uninstall(models_dir):
    """atexit handler: silently restore originals + clean .pyc."""
    try:
        p = _resolve_paths(models_dir)
        for fpath in p["all_files"]:
            bak = fpath + BACKUP_SUFFIX
            if os.path.isfile(bak):
                shutil.copy2(bak, fpath)
                os.remove(bak)
        _clean_pycache(p["cache_dirs"], verbose=False)
        print("[PATCH] ✓ Restored original sglang source files.",
              file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PATCH] WARNING: auto-restore failed: {e}",
              file=sys.stderr, flush=True)
        print("[PATCH] Run manually:  python patch_sglang_all_layers.py --uninstall",
              file=sys.stderr, flush=True)
    finally:
        # clear the patched flag so a later auto_patch() in this process can run install again
        os.environ.pop(_ENV_PATCHED, None)


def auto_patch(layers=None):
    """
    Auto-patch sglang source files.  Call BEFORE ``import sglang``.

    Args:
        layers: Which layer representations to capture.
            - None or "all": all layers + final norm (default)
            - str: comma-separated indices, e.g. "0,5,28"
            - list/tuple of int: e.g. [0, 5, 28]
          Index semantics (N = num_hidden_layers):
            0       = embedding output
            1..N-1  = after transformer layers 0..N-2
            N       = final RMSNorm output

    Checks the SGLang version, then backs up and patches the registered model,
    processor, scheduler, and backend files. Patches remain installed until
    --uninstall or auto_uninstall() is called explicitly.

    Spawned workers inherit the environment marker and skip re-patching.
    Independent processes use file locking to avoid concurrent writes.
    """
    import fcntl

    # Spawned workers inherit the parent's patches and skip re-patching.
    if os.environ.get(_ENV_PATCHED) == "1":
        print("[PATCH] Already patched — skipping.", file=sys.stderr, flush=True)
        return

    # Set layer capture configuration (inherited by spawn'd children)
    if layers is not None:
        if isinstance(layers, (list, tuple)):
            os.environ[_ENV_CAPTURE_LAYERS] = ",".join(str(x) for x in layers)
        else:
            os.environ[_ENV_CAPTURE_LAYERS] = str(layers)
    spec = os.environ.get(_ENV_CAPTURE_LAYERS, "all")
    print(f"[PATCH] Capture layers: {spec}", file=sys.stderr, flush=True)

    _check_version()

    models_dir = find_sglang_models_dir()
    version = _detected_version

    # Use file lock to prevent race conditions when multiple processes patch simultaneously
    lock_file = os.path.join(models_dir, ".sglang_patch.lock")

    try:
        # Create lock file if not exists
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)

        # Acquire exclusive lock (blocking)
        print("[PATCH] Acquiring lock...", file=sys.stderr, flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print("[PATCH] Lock acquired.", file=sys.stderr, flush=True)

        try:
            # Now safe to patch (other processes will wait for lock)
            install(models_dir, version=version)
        finally:
            # Release lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    except (OSError, IOError) as e:
        # If locking fails (e.g., unsupported filesystem), fall back to original behavior
        print(f"[PATCH] Warning: File locking failed ({e}), proceeding without lock",
              file=sys.stderr, flush=True)
        install(models_dir, version=version)

    # Mark so spawn'd children skip
    os.environ[_ENV_PATCHED] = "1"


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true",
                       help="Backup originals and apply patches")
    group.add_argument("--uninstall", action="store_true",
                       help="Restore originals from backup")
    group.add_argument("--status", action="store_true",
                       help="Show current patch status")
    parser.add_argument("--models-dir", default=None,
                        help="Path to sglang/srt/models/ dir (auto-detected if omitted)")
    parser.add_argument("--skip-version-check", action="store_true",
                        help="Skip sglang version check")
    args = parser.parse_args()

    if not args.skip_version_check and (args.install or args.status):
        version = _check_version()
    else:
        version = _detected_version

    models_dir = args.models_dir or find_sglang_models_dir()

    if args.install:
        install(models_dir, version=version)
    elif args.uninstall:
        uninstall(models_dir)
    else:
        status(models_dir, version=version)


if __name__ == "__main__":
    main()
