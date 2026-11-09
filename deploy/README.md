## Inference

PGuard serves a trained safety head on top of a frozen backbone inside
[SGLang](https://github.com/sgl-project/sglang). The head consumes hidden states
captured from a few selected backbone layers and emits per-class risk scores
alongside the generated tokens, so scoring costs no extra forward pass and adds
no extra request.

### Install

```bash
# 0.5.11 for the Qwen3.5 and GLM-4.5V
pip install -e '.[sglang0511]'
# 0.5.17 for DeepSeek-V4-Flash
pip install -e '.[sglang0517]'
```

### Serving

The package registers its models through SGLang's external-model hook:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=pguard.sglang
export PGUARD_OVERWRITE_MODEL=1
export PGUARD_MODELING_VARIANT=seq          # stateless | seq | cfc, required
export PGUARD_SEQ_STATE_PATH=/path/to/head

python -m sglang.launch_server \
    --model-path /path/to/Qwen3.5-9B \
    --tp-size 1 \
    --trust-remote-code \
    --incremental-streaming-output
```

### Interrupt policy (Optional)

Add `--interrupt_config /path/to/policy.json` to stop a request when a scored
category exceeds its threshold. Rules are stage-specific:

```json
[
  "prefill.Crimes & Illegal Activities###Dangerous Weapons@0.9",
  "decode.Crimes & Illegal Activities###Dangerous Weapons@0.95"
]
```

### Consuming the scores

Per-head softmax scores are returned in `meta_info.guard_logits`.
Each token has a class-score vector on scoring steps, or `0.0` otherwise.

```python
chunk["meta_info"]["guard_logits"]   # [[v1, ..., vC]] scored this token
                                     # [0.0]            did not score this token
```

### Variants

| Variant | Head version | Weight env | Scoring |
|---|---|---|---|
| `stateless` | `lasttok` | `PGUARD_MODEL_PATH` | Last token only, no state carried between steps |
| `seq` | `ema` | `PGUARD_SEQ_STATE_PATH` | Per token, over multi-scale EMA history |
| `cfc` | `gru` | `PGUARD_CFC_STATE_PATH` | Per token, query pooling plus a response CfC-GRU |

### Supported backbones

| Backbone | `model_type` | Entry module | SGLang |
|---|---|---|---|
| Qwen3.5-9B | `qwen3_5` | `qwen3_5_pguard_sglang` | 0.5.11 |
| Qwen3.5-35B-A3B | `qwen3_5_moe` | `qwen3_5_pguard_sglang` | 0.5.11 |
| Qwen3.5-122B-A10B | `qwen3_5_moe` | `qwen3_5_pguard_sglang` | 0.5.11 |
| GLM-4.5V | `glm4v_moe` | `glm4v_moe_pguard_sglang` | 0.5.11 |
| DeepSeek-V4-Flash | `deepseek_v4` | `deepseek_v4_pguard_sglang` | 0.5.17 |


### Defaults

Common:

| Variable | Default | Meaning |
|---|---|---|
| `PGUARD_MODELING_VARIANT` | — | Required: `stateless`, `seq` or `cfc` |
| `PGUARD_OVERWRITE_MODEL` | `0` | `1` replaces the native classes with the guard ones |
| `PGUARD_COPY_LOGITS_PROB` | `0.1` | Ratio of scoring |
| `PGUARD_CAPTURE_LAYERS` | config `pguard_capture_layers` | Absolute backbone layers to capture; overrides the config if set |
| `PGUARD_SKIP_FIRST_TOKEN` | `0` | `1` skips scoring the prefill-produced token |
| `PGUARD_ASYNC_COPY` | `0` | `1` copies scores device-to-host asynchronously |
| `PGUARD_BREAKABLE_GRAPH` | `0` | `1` lets non-scoring steps skip the guard CUDA graph segment |

`seq`:

| Variable | Default | Meaning |
|---|---|---|
| `PGUARD_EMA_SCALES` | `200,150,100,50` | EMA window lengths in tokens |
| `PGUARD_PREFILL_SCORE_PROB` | `1.0` | Ratio of scoring |
| `PGUARD_PREFILL_RISK_MODE` | `last` | `last`: final position; `all`: per-class max over scored positions, without renormalization |
| `PGUARD_MAX_REQ` | `256` | History buffer fallback size |

`cfc`:

| Variable | Default | Meaning |
|---|---|---|
| `PGUARD_CFC_PROJ_DIM` | `768` | CfC input projection width |
| `PGUARD_CFC_MEM_DIM` | `768` | Recurrent state width |
| `PGUARD_CFC_NUM_CLASSES` | `4` | Used only when the checkpoint carries no task schema |
| `PGUARD_CFC_PREFILL_MODE` | `pool` | `pool` or `ema` prompt summarisation |
| `PGUARD_CFC_PREFILL_EFFECTIVE_LENGTHS` | `200,150,100,50` | EMA window lengths under `ema` |
