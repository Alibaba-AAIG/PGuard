#!/usr/bin/env bash
# Multi-version acceptance via pip swaps inside the running guard container.
#
# usage: bash tests/versions/run_matrix_pip.sh [version ...]
#   no args: every vLLM release on PyPI >= MIN_VERSION (default 0.25.0)
# env:  MIN_VERSION, PIP_INDEX_URL (pip index; default Tsinghua mirror)
#       WHEEL_FLAVORS (cuda flavors to try, default "cu129"; versions without
#       a listed flavor are skipped -> run them via the image matrix instead)
#
# torch / cuda / nvidia runtime stay untouched: each version installs the
# +cu129 wheel variant built against the container's existing CUDA 12.9
# stack (index: https://wheels.vllm.ai/<ver>/<flavor>/), ~400 MB per version.
#
# Per version: pip install in the container -> qwen3_5 gate -> restart ->
# compat/smoke/interrupt/cadence. A crashed engine is rescued by recreating
# the container from the image (pristine baseline). The sweep ends with the
# same recreate, so the container always lands back on the image's baseline.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONTAINER=pguard-vllm-guard
PORT="${PGUARD_PORT:-13553}"
MIN_VERSION="${MIN_VERSION:-0.25.0}"
# pip index defaults to the Tsinghua mirror; override with PIP_INDEX_URL.
# docker exec does not inherit host env, so it is forwarded explicitly.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
WHEEL_FLAVORS="${WHEEL_FLAVORS:-cu129}"   # CACHE_LIMIT_GB: wheel stock cap (default 8 GB)
SUMMARY="tests/versions/results/SUMMARY-pip.md"
mkdir -p tests/versions/results
touch "$SUMMARY"

versions() {
  if [ $# -gt 0 ]; then printf '%s\n' "$@"; return; fi
  curl -s --max-time 30 https://pypi.org/pypi/vllm/json | python3 -c "
import json, sys, re
d = json.load(sys.stdin)
key = lambda v: tuple(int(x) for x in re.findall(r'\d+', v)[:3])
vers = {v for v in d['releases'] if re.match(r'^0\.\d+(\.\d+)?$', v)}
lo = [v for v in vers if key(v) >= tuple(int(x) for x in '$MIN_VERSION'.split('.'))]
print('\n'.join(sorted(lo, key=key)))"
}

wait_healthy() {
  for _ in $(seq 1 36); do
    s=$(curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)
    echo "$s" | grep -q '"ok"' && return 0
    sleep 10
  done
  return 1
}

record() { echo "| $1 | $2 | $3 |" >> "$SUMMARY"; }

restore() {  # recreate from the image: pristine baseline vLLM
  docker compose --profile vllm-guard up -d --force-recreate "$CONTAINER" \
    && wait_healthy
}

run_step() {  # run_step <ver> <name> <cmd...>
  local ver="$1" name="$2"; shift 2
  if "$@" > "tests/versions/results/pip-$ver-$name.log" 2>&1; then
    record "$ver" "$name" PASS; return 0
  fi
  record "$ver" "$name" FAIL; return 1
}

echo "## pip run $(date '+%F %T') versions: ${*:-all >= $MIN_VERSION}" >> "$SUMMARY"

for ver in $(versions "$@"); do
  echo "=== vLLM $ver ==="
  flavor=""
  for f in $WHEEL_FLAVORS; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
      "https://wheels.vllm.ai/$ver/$f/")
    [ "$code" = "200" ] && { flavor="$f"; break; }
  done
  if [ -z "$flavor" ]; then
    record "$ver" "cuda wheel" "SKIP (no $WHEEL_FLAVORS index -> image matrix)"
    continue
  fi
  # torch / cuda / nvidia runtime stay untouched: +cu129 wheel variants are
  # built against the container's existing CUDA 12.9 stack
  if ! docker exec -e PIP_INDEX_URL="$PIP_INDEX_URL" "$CONTAINER" \
      pip install -q --extra-index-url "https://wheels.vllm.ai/$ver/$flavor" \
      "vllm==$ver+$flavor"; then
    record "$ver" "pip install" FAIL
    restore || record "$ver" "restore" FAIL   # a half-installed vllm breaks imports
    continue
  fi
  # heal partial upgrades: a vllm swap can leave image-pinned extras
  # (e.g. flashinfer vs flashinfer-cubin) mismatched, killing the import
  docker exec -e PIP_INDEX_URL="$PIP_INDEX_URL" "$CONTAINER" sh -c \
    'pip show flashinfer >/dev/null 2>&1 && pip install -q --upgrade flashinfer flashinfer-cubin flashinfer-python' \
    || true
  if ! docker exec "$CONTAINER" python3 -c \
      "import importlib.util, os, sys
spec = importlib.util.find_spec('vllm')  # top-level lookup executes nothing
root = os.path.dirname(spec.origin)
sys.exit(0 if os.path.exists(os.path.join(root, 'model_executor', 'models', 'qwen3_5.py')) else 1)"; then
    record "$ver" "qwen3_5 arch" SKIP; continue
  fi
  docker restart "$CONTAINER" >/dev/null
  if ! wait_healthy; then
    docker logs "$CONTAINER" --since 12m 2>&1 | tail -30 \
      > "tests/versions/results/pip-$ver-engine.log"
    if grep -q "libcudart" "tests/versions/results/pip-$ver-engine.log"; then
      record "$ver" "healthy" "FAIL (cuda runtime mismatch)"
    else
      record "$ver" "healthy" "FAIL (engine crash)"
    fi
    restore || record "$ver" "restore" FAIL
    continue
  fi
  run_step "$ver" compat    docker exec "$CONTAINER" python3 -m pguard_vllm.vllm_compat
  run_step "$ver" smoke     bash tests/pguard_smoke_test.sh "$PORT"
  run_step "$ver" interrupt python3 tests/pguard_interrupt_e2e.py "http://127.0.0.1:$PORT" 6
  run_step "$ver" cadence   python3 tests/pguard_cadence_e2e.py "http://127.0.0.1:$PORT" 8
  # wheel stock cap: purge when the cache exceeds ~2 versions' worth
  used_gb=$(docker exec "$CONTAINER" sh -c \
    "du -s -m /root/.cache/pip 2>/dev/null" | awk '{print int($1/1024)}')
  if [ "${used_gb:-0}" -gt "${CACHE_LIMIT_GB:-8}" ]; then
    docker exec -e PIP_INDEX_URL="$PIP_INDEX_URL" "$CONTAINER" \
      pip cache purge >/dev/null && echo "cache purged (${used_gb}GB in stock)"
  fi
done

echo "=== restoring baseline from image ==="
if restore; then
  record baseline "restore ($(docker exec "$CONTAINER" python3 -c \
    'import vllm; print(vllm.__version__)' 2>/dev/null))" "OK"
else
  record baseline "restore" FAIL
fi
echo "=== summary in $SUMMARY ==="
tail -n 20 "$SUMMARY"
