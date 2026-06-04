#!/usr/bin/env bash
# gpumod-h6gs benchmark driver: runs gemma4-12b-q4 then gemma4-12b-q5
# sequentially. Each step starts the service, waits for /health, runs the
# 15-iter v2 coding benchmark, stops the service, and waits past the
# quiesce window before moving on.
#
# Launch inside a tmux session ('bench') and pair with a 'monitor' session
# per .claude/CLAUDE.md "Running Long Benchmarks".
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp"
QUIESCE_SECS=20

run_one() {
    local model="$1" port="$2"
    echo "=== $(date -Iseconds) starting service $model on port $port ==="
    uv run gpumod service start "$model"
    echo "=== waiting for /health on $port ==="
    local waited=0
    until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        sleep 3
        waited=$((waited + 3))
        if [ "$waited" -gt 300 ]; then
            echo "!!! $model failed to come up in 5min — aborting"
            uv run gpumod service stop "$model" || true
            exit 1
        fi
    done
    echo "=== health OK after ${waited}s, launching benchmark ==="
    uv run python scripts/run_qwen36_benchmark.py \
        --model "$model" \
        --output-dir "$OUT" \
        2>&1 | tee -a "$OUT/run_${model}.log"
    echo "=== $(date -Iseconds) benchmark $model done, stopping service ==="
    uv run gpumod service stop "$model"
    echo "=== quiescing ${QUIESCE_SECS}s before next model ==="
    sleep "$QUIESCE_SECS"
}

run_one gemma4-12b-q4 7106
run_one gemma4-12b-q5 7107

echo "=== $(date -Iseconds) ALL BENCHMARKS COMPLETE ==="
