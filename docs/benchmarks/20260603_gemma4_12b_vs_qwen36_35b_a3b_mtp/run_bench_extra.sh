#!/usr/bin/env bash
# gpumod-h6gs extension: runs gemma4-12b-q8 then gemma4-26b-a4b-q4 sequentially.
# Launch AFTER the original run_bench.sh completes (when both Q4 and Q5 are
# done). Pair with a 'monitor' tmux session per .claude/CLAUDE.md.
#
# Pre-flight: 26B-A4B model file must finish downloading before this script
# starts; the script checks and aborts if it looks truncated.
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp"
QUIESCE_SECS=20

# 26B-A4B file size guard: expect ~12.7 GiB (13597175584 bytes per HF manifest).
# Refuse to start if under 12 GiB — likely still downloading.
MODEL_26B="$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
if [ ! -f "$MODEL_26B" ]; then
    echo "!!! $MODEL_26B not found"
    exit 1
fi
SIZE_BYTES=$(stat -c%s "$MODEL_26B")
MIN_BYTES=$((12 * 1024 * 1024 * 1024))
if [ "$SIZE_BYTES" -lt "$MIN_BYTES" ]; then
    echo "!!! $MODEL_26B is only $((SIZE_BYTES / 1024 / 1024)) MiB — expected 12+ GiB. Download likely still in progress."
    exit 1
fi

# VRAM isolation per .claude/CLAUDE.md "Running Long Benchmarks": stop all
# gpumod-tracked services before the first model start. Idempotent — if mode
# is already blank, this is a no-op.
echo "=== $(date -Iseconds) ensuring blank mode (VRAM isolation) ==="
uv run gpumod mode switch blank

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

run_one gemma4-12b-q8 7108
run_one gemma4-26b-a4b-q4 7109

echo "=== $(date -Iseconds) ALL EXTRA BENCHMARKS COMPLETE ==="
