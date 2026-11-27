# PGuard × vLLM serving integration

Serve a trained PGuard safety head on top of a frozen backbone inside
[vLLM](https://github.com/vllm-project/vllm) (tested with v0.25.1). The head
consumes hidden states captured from a few selected backbone layers and emits
per-class risk scores alongside the generated tokens, so scoring costs no
extra forward pass and adds no extra request. **No vLLM source changes and no
weight changes** -- the integration is a pip-installable package that hooks in
through vLLM's `vllm.general_plugins` entry point.

Verified with Qwen3.5-9B (FP8-dynamic) on a single RTX 4090 24G: three heads
online, 8-way concurrency at **328–350 tok/s** (pure-backbone eager baseline:
169–210 tok/s), standalone moderation round-trip ~64 ms. With scoring thinned
to every 10th output token (`PGUARD_COPY_LOGITS_PROB=0.1`) the same setup
reaches **463–468 tok/s**.

## Install

```bash
pip install --no-build-isolation --no-deps -e ./pguard_vllm
```

The package registers itself through the `vllm.general_plugins` entry point:
`pguard_vllm.plugin:register` installs the custom architecture
(`Qwen3_5PGuardForConditionalGeneration`) into vLLM's `ModelRegistry` and
patches the GPU model runner, so the original backbone checkpoint loads
unchanged.

## Serving

```bash
export PGUARD_HEADS=PGuard-EMA-Qwen3.5-9B      # checkpoint dir name(s) under /pguard_heads
export PGUARD_ENFORCE_EAGER=0                  # CUDA graphs on
export PGUARD_CAPTURE_MODE=aux                 # EAGLE3-style layer capture
export VLLM_USE_V2_MODEL_RUNNER=1              # V2 runner supports aux tuples

python3 -m pguard_vllm.server \
    --model /path/to/Qwen3.5-9B-FP8-dynamic \
    --port 13553 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --max-num-seqs 32
```

Or via Docker Compose (see `docker-compose.yml`).

## Endpoints

- `POST /v1/chat/completions` -- OpenAI-compatible generation; non-stream
  responses carry `guard_result`, and `"stream": true, "guard_interrupt": true`
  aborts as soon as a scored category crosses the threshold. A prompt-side
  hit stops the stream before any content chunk is emitted (verified 30/30
  trials on jailbreak-style prompts).
- `POST /v1/guard/score` -- standalone moderation of a message list (no
  generation), ~64 ms round-trip.
- `GET /v1/models`, `GET /health`.

## How it works

1. The entry point registers `Qwen3_5PGuardForConditionalGeneration` in
   vLLM's `ModelRegistry`; `--hf-overrides` (set by the server) makes vLLM
   load the original backbone checkpoint into this subclass.
2. `_setup_pguard()` loads the safety-head checkpoints and captures layers
   20/24/28 either through forward hooks (eager) or through vLLM's
   EAGLE3-style aux-hidden-state mechanism (CUDA-graph safe; the
   `use_aux_hidden_state_outputs` flag is flipped after `load_model` to skip
   the drafter assertion).
3. After each engine step, the patched `execute_model` segments the captured
   hidden states per request (via `SchedulerOutput.num_scheduled_tokens`) and
   runs the heads: batched scoring, one device-to-host transfer per head, flat
   task-major score records.
4. Scores land in a file-backed store; the API aggregates them (running
   max/last maintained on the GPU, finalized once per request) and attaches
   `guard_result` to responses. The engine also keeps a tiny per-request
   live status file (final prompt verdict + running response max) that the
   streaming interrupt check reads, so later output records never mask a
   prompt-side hit.

See `DESIGN.md` for the vLLM-internals facts this relies on.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PGUARD_HEADS` | -- | comma-separated checkpoint dir names (multi-head) |
| `PGUARD_HEAD_PATH` | -- | single-head checkpoint dir |
| `PGUARD_THRESHOLD` | `0.5` | non-normal class hit threshold |
| `PGUARD_ENFORCE_EAGER` | `0` | `1` disables CUDA graphs (hooks mode requires eager) |
| `PGUARD_CAPTURE_MODE` | `aux` | `aux` (graph-safe) or `hooks` (eager only) |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | use the V2 GPU model runner |
| `PGUARD_COPY_LOGITS_PROB` | `1.0` | scoring cadence; `0.1` scores every 10th output token (prefill always scores) |
| `PGUARD_SCORE_INTERVAL` | -- | scoring interval in steps (overrides the ratio) |
| `PGUARD_PREFILL_RISK_MODE` | `all` | `all` scores every prefill token; `last` scores the final token only |
| `PGUARD_HEAD_DTYPE` | `fp32` | head compute dtype (`bf16` halves head bandwidth) |
| `PGUARD_FUSE_OPS` | `1` | fused classifier/adapter/GRU-gate kernels |
| `PGUARD_MAX_SLOTS` | `64` | slot-pool capacity (>= max_num_seqs) |
| `PGUARD_CFC_PREFILL_MODE` | `pool` | GRU-head prefill summarization: `pool` or `ema` |
| `PGUARD_STORE_DIR` | `/dev/shm/pguard_store` | score store directory |

## Testing

```bash
# module self-tests
python3 -m pguard_vllm.store
python3 -m pguard_vllm.state_pool
python3 -m pguard_vllm.pguard_config        # env config parsing
python3 -m pguard_vllm.model        # scoring-cadence dispatch
python3 -m pguard_vllm.server --self-test   # streaming interrupt check
python3 -m pguard_vllm.heads        # needs PGUARD_PW_DIR pointing at checkpoints

# end-to-end verdict cases
bash tests/pguard_smoke_test.sh 13553

# streaming interruption: risky prompts stop with zero content chunks,
# benign prompts finish normally
python3 tests/pguard_interrupt_e2e.py http://127.0.0.1:13553 10

# late-arriving prompts keep their prompt-side scores under scoring cadence
python3 tests/pguard_cadence_e2e.py http://127.0.0.1:13553 16

# concurrency benchmark
python3 tests/bench_guard.py http://127.0.0.1:13553 8
```

## vLLM version compat

| Version | Status |
|---|---|
| v0.25.1 | verified (eager + CUDA graph, 3 heads, multimodal) |
| v0.34.0 | design-time source check (runner / SchedulerOutput fields match) |
| others  | run `python3 -m pguard_vllm.vllm_compat` in the target env, extend `PGUARD_RUNNER_CANDIDATES` / `vllm_compat.py` until green, then run the acceptance flow |

Runner classes, `SchedulerOutput` fields and aux hidden-state storage are
probed in one place (`pguard_vllm/vllm_compat.py`); `python3 -m
pguard_vllm.vllm_compat` prints the report and exits non-zero if the patch
cannot attach.

## Known limitations

1. Speculative decoding is not supported (conflicts with per-request scoring).
2. CUDA-graph mode shifts scores by ~1% relative to eager (compile + padding
   numerics); verdicts are unchanged.
3. FP8 backbones vs bf16-trained heads can shift scores; calibrate thresholds
   on business samples. The GRU head is the least discriminative under FP8.
4. The EMA head's prompt series is per-prefill-chunk (last token); per-token
   series for EMA requires the sampled-EMA kernel (planned).
5. With `PGUARD_COPY_LOGITS_PROB` < 1 only 1/N output tokens are scored:
   `response_scores_max` covers scored tokens only, and a response-side
   streaming interrupt can trail by up to N-1 tokens (the prompt side is
   unaffected).
6. Multimodal image input is supported via OpenAI content parts (`text` +
   `image_url`, data: base64 or http(s) URLs; `PGUARD_MM_LIMIT` caps images
   per prompt, default 4). The guard scores the fused vision/text hidden
   states per token, same as the official GLM4V integration. Audio/video
   inputs are not covered.
