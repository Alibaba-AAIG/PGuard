# Tests & acceptance

Service-level tests assume the server is running (`docker compose --profile
vllm-guard up -d`, default port 13553; port 8000 inside the container).
Module-level self-tests need no HTTP service and run inside the container.

## Service-level acceptance

Prerequisite: `curl http://127.0.0.1:13553/health` returns `{"status":"ok"}`.

| Script | What it checks | Command |
|---|---|---|
| `pguard_smoke_test.sh` | 5 verdict smoke cases: health, malicious/benign standalone scoring, normal/induced chat with `guard_result`. JSON results in `tests/artifacts/` | `bash tests/pguard_smoke_test.sh 13553` |
| `pguard_interrupt_e2e.py` | Streaming interrupt correctness: risky requests must be cut BEFORE any content chunk (zero leakage); benign requests must finish normally | `python3 tests/pguard_interrupt_e2e.py http://127.0.0.1:13553 10` |
| `pguard_cadence_e2e.py` | Scoring cadence: late-joining requests keep their prompt-side scores; output tokens follow the cadence | `python3 tests/pguard_cadence_e2e.py http://127.0.0.1:13553 16` |
| `bench_guard.py` | Performance: 8-way concurrency throughput / TTFT / single-stream speed (compare with the README table) | `python3 tests/bench_guard.py http://127.0.0.1:13553 8` |

## Module self-tests (in-container, no HTTP)

```bash
docker exec pguard-vllm-guard python3 -m pguard_vllm.vllm_compat
docker exec pguard-vllm-guard python3 -m pguard_vllm.pguard_config
docker exec pguard-vllm-guard python3 -m pguard_vllm.store
docker exec pguard-vllm-guard python3 -m pguard_vllm.state_pool
docker exec pguard-vllm-guard python3 -m pguard_vllm.model
docker exec pguard-vllm-guard python3 -m pguard_vllm.server --self-test
docker exec -e PGUARD_PW_DIR=/pguard_heads pguard-vllm-guard python3 -m pguard_vllm.heads
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.lasttok
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.ema
docker exec pguard-vllm-guard python3 -m pguard_vllm.heads.gru
```

## Multi-version matrix

Run the same acceptance per vLLM version: `tests/versions/` (list in
`versions.py`, entry point `run_matrix.sh`, results in
`tests/versions/results/`).

```bash
bash tests/versions/run_matrix.sh              # all versions in the list
PULL=1 bash tests/versions/run_matrix.sh 0.26.0   # pull first, then test
```

## Upgrade / re-model acceptance flow

After a vLLM upgrade, a backbone swap, or any patch change, run in order and
stop on the first failure:

0. in the TARGET environment: `python3 -m pguard_vllm.vllm_compat` — the
   patch-dependency report must be green (extend `PGUARD_RUNNER_CANDIDATES`
   / `vllm_compat.py` otherwise)
1. module self-tests (all green)
2. start the service, wait for healthy
3. `pguard_smoke_test.sh` — verdicts must match the baseline
4. `pguard_interrupt_e2e.py` — zero leakage must hold
5. `pguard_cadence_e2e.py`
6. `bench_guard.py` — compare with the README table (±10% is normal variance)
