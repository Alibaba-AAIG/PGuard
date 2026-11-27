PR: https://github.com/Alibaba-AAIG/PGuard/pull/1
（标题 GitHub 已自动取提交主题："Add vLLM serving integration (deploy/vllm)"，无需改；只需把下面正文粘进 PR 的描述框：PR 页面右上角 "…" → Edit → 粘贴 → Update）

---------- 以下为 PR 正文 ----------

## What

A serving integration that runs a trained PGuard safety head on top of a frozen
backbone inside [vLLM](https://github.com/vllm-project/vllm) (tested with
v0.25.1), mirroring the SGLang integration in `deploy/`:

- OpenAI-compatible chat whose responses carry per-step `guard_result`
  (9 risk dimensions, 38 classes), with streaming risk interruption
  (`guard_interrupt`) and a standalone moderation endpoint `POST /v1/guard/score`
  (~64 ms round-trip).
- Hidden states of backbone layers 20/24/28 are captured and scored by the
  safety head **during generation -- no extra forward pass, no extra request**.
- `pguard_vllm` pip package registered through vLLM's `vllm.general_plugins`
  entry point; **no vLLM source changes, no weight changes** -- the original
  backbone checkpoint loads unchanged into a registered subclass.
- Supports lasttok / EMA / GRU heads, multi-head simultaneously.

## Performance (Qwen3.5-9B FP8, RTX 4090 24G, 8-way concurrency x 256 tokens, 3 heads online)

| Configuration | Throughput | Single-stream |
|---|---|---|
| pure backbone, eager | 169~210 tok/s | 24.0 chunks/s |
| + guard, eager (fused ops, flat records, slot-pooled state) | 152~178 tok/s | 24.6~25.7 chunks/s |
| + guard, **CUDA graphs (V2 runner + EAGLE3-style aux capture)** | **328~350 tok/s** | **52.7~55.9 chunks/s** |
| + guard, CUDA graphs, scoring every 10th output token (`PGUARD_COPY_LOGITS_PROB=0.1`) | **463~468 tok/s** | **65.3 chunks/s** |
| pure backbone, CUDA graphs (reference) | 505 tok/s | 69.8 chunks/s |

Standalone moderation: ~64 ms round-trip.

## How it works

1. `vllm.general_plugins` -> `pguard_vllm.plugin:register` registers
   `Qwen3_5PGuardForConditionalGeneration` in the `ModelRegistry` and patches
   the GPU model runner; `--hf-overrides` makes vLLM load the original
   backbone checkpoint into this subclass.
2. Backbone layers 20/24/28 are captured with vLLM's EAGLE3-style aux
   hidden-state mechanism (`set_aux_hidden_state_layers`); the
   `use_aux_hidden_state_outputs` flag is flipped **after** `load_model` so the
   V2 runner's drafter assertion is not triggered. CUDA graphs (FULL for
   decode, PIECEWISE for prefill) capture and replay normally.
3. After each engine step, the patched `execute_model` segments the captured
   hidden states per request using `SchedulerOutput.num_scheduled_tokens` and
   runs the heads: batched scoring with fused classifier/adapter/GRU-gate ops,
   closed-form EMA prefill folding, slot-pooled recurrent state, and one
   device-to-host transfer per head producing flat task-major score records.
4. Per-request running max/last are maintained on the GPU; when the scheduler
   reports the request finished, a single `final` record is written, so the
   API reads results in O(1).
5. Scoring cadence follows the SGLang integration (`PGUARD_COPY_LOGITS_PROB`);
   prefill rows always score and the recurrent state updates every step, only
   the output-side scoring is thinned.
6. The engine keeps a tiny per-request live status file (final prompt verdict
   + running response max, rewritten only when the request's top non-normal
   probability rises) that the streaming interrupt check reads, so later
   output records never mask a prompt-side hit.

## Test

- Module self-tests: `python3 -m pguard_vllm.store`,
  `python3 -m pguard_vllm.state_pool`,
  `python3 -m pguard_vllm.model` (scoring-cadence dispatch),
  `python3 -m pguard_vllm.server --self-test` (streaming interrupt check),
  `python3 -m pguard_vllm.head` (against real checkpoints).
- End-to-end verdict cases: `pguard_smoke_test.sh` (5 cases); streaming
  interruption: `pguard_interrupt_e2e.py` -- 3 jailbreak-style prompts x 10
  trials, 30/30 `content_filter` with zero content chunks emitted before the
  stop; late-arriving prompts under scoring cadence: `pguard_cadence_e2e.py`
  (16/16 keep their prompt-side scores).
- Concurrency benchmark: `bench_guard.py`.
- Multi-head agreement verified on jailbreak-style prompts; 0 scoring
  failures across runs.
