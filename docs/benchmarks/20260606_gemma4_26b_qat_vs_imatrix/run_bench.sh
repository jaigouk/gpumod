#!/usr/bin/env bash
# gpumod-rjkx benchmark driver: Gemma 4 26B-A4B QAT UD-Q4_K_XL.
#
# Runs the QAT arm only — the imatrix baseline reuses the existing
# result_gemma4-26b-a4b-q4.json from docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/
# (same b9500 binary, same harness, same GEMMA_CODING sampler, same single-slot
# multimodal config). Methodology caveat: different host-load window — the
# README must flag this explicitly when reporting TPS deltas.
#
# Launch inside a tmux 'bench' session and pair with a 'monitor' session per
# .claude/CLAUDE.md "Running Long Benchmarks".
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="docs/benchmarks/20260606_gemma4_26b_qat_vs_imatrix"
QUIESCE_SECS=20

# VRAM isolation per .claude/CLAUDE.md: stop all gpumod-tracked services
# before the first model start. Idempotent — if mode is already blank,
# this is a no-op.
echo "=== $(date -Iseconds) ensuring blank mode (VRAM isolation) ==="
uv run gpumod mode switch blank

# Wait past the quiesce window (10s configured) before launching the
# benchmark target. Otherwise a stop here triggers a `service start` failure
# below with "Quiesce period active: 9s remaining". 15s is conservative.
echo "=== $(date -Iseconds) quiesce wait (15s) ==="
sleep 15

archive_prior() {
    # Archive any existing result/log/artifacts for $model to the next
    # available .runN slot before launching a fresh run. Idempotent — if
    # no current data exists (fresh-state launch), the function returns
    # without renaming anything.
    local model="$1"
    if [ ! -f "$OUT/result_${model}.json" ]; then
        return 0
    fi
    local n=1
    while [ -e "$OUT/result_${model}.run${n}.json" ] \
       || [ -e "$OUT/run_${model}.run${n}.log" ] \
       || [ -e "$OUT/artifacts/${model}.run${n}" ]; do
        n=$((n + 1))
    done
    echo "=== $(date -Iseconds) archiving current ${model} data → run${n} ==="
    mv "$OUT/result_${model}.json" "$OUT/result_${model}.run${n}.json"
    [ -f "$OUT/run_${model}.log" ]   && mv "$OUT/run_${model}.log"   "$OUT/run_${model}.run${n}.log"
    [ -d "$OUT/artifacts/${model}" ] && mv "$OUT/artifacts/${model}" "$OUT/artifacts/${model}.run${n}"
}

run_one() {
    local model="$1" port="$2"
    archive_prior "$model"
    echo "=== $(date -Iseconds) starting service $model on port $port ==="
    if ! uv run gpumod service start "$model"; then
        echo "!!! service start failed for $model — aborting"
        exit 1
    fi
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

# QAT only. Imatrix baseline is reused from the 20260603 bench folder.
run_one gemma4-26b-a4b-qat-q4 7110

echo "=== $(date -Iseconds) QAT BENCHMARK COMPLETE ==="
