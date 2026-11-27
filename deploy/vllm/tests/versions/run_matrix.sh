#!/usr/bin/env bash
# Multi-version acceptance matrix.
#
# usage: bash tests/versions/run_matrix.sh [version ...]
#   default: every entry in tests/versions/versions.py
# env:   PULL=1  docker pull the image before running
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p tests/versions/results

PORT="${PGUARD_PORT:-13553}"
FILTERS="$*"
SUMMARY="tests/versions/results/SUMMARY.md"
touch "$SUMMARY"
echo "## run $(date '+%F %T') versions: ${FILTERS:-all}" >> "$SUMMARY"

versions() {
  python3 tests/versions/versions.py
}

wait_healthy() {
  for _ in $(seq 1 60); do
    s=$(curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)
    echo "$s" | grep -q '"ok"' && return 0
    sleep 20
  done
  return 1
}

run_step() {  # run_step <ver> <name> <cmd...>
  local ver="$1" name="$2"; shift 2
  if "$@" > "tests/versions/results/$ver-$name.log" 2>&1; then
    echo "| $ver | $name | PASS |" >> "$SUMMARY"; return 0
  fi
  echo "| $ver | $name | FAIL |" >> "$SUMMARY"; return 1
}

while read -r ver image; do
  [ -z "${ver:-}" ] && continue
  if [ -n "$FILTERS" ]; then
    case " $FILTERS " in *" $ver "*) ;; *) continue;; esac
  fi
  echo "=== $ver ($image) ==="
  if [ "${PULL:-0}" = "1" ]; then
    docker pull "$image" || { echo "| $ver | pull | FAIL |" >> "$SUMMARY"; continue; }
  fi
  if ! VLLM_IMAGE="$image" docker compose --profile vllm-guard up -d \
      --force-recreate pguard-vllm-guard; then
    echo "| $ver | compose up | FAIL |" >> "$SUMMARY"; continue
  fi
  if ! wait_healthy; then
    echo "| $ver | healthy | FAIL (timeout) |" >> "$SUMMARY"
    docker logs pguard-vllm-guard --since 15m 2>&1 | tail -30 \
      > "tests/versions/results/$ver-engine.log"
    continue
  fi
  run_step "$ver" compat  docker exec pguard-vllm-guard python3 -m pguard_vllm.vllm_compat
  run_step "$ver" smoke   bash tests/pguard_smoke_test.sh "$PORT"
  run_step "$ver" interrupt python3 tests/pguard_interrupt_e2e.py "http://127.0.0.1:$PORT" 6
  run_step "$ver" cadence python3 tests/pguard_cadence_e2e.py "http://127.0.0.1:$PORT" 8
done < <(versions)

echo "=== summary in $SUMMARY ==="
tail -n 8 "$SUMMARY"
