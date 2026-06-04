#!/usr/bin/env bash
# gpumod-8xaq Phase 2 — one-shot N=3 ctx=200K Workflow A bench (text-only).
# Same safeguards as the main orchestrator: GGML_CUDA_NO_PINNED, RAM/VRAM
# preflight, watchdog.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260604_multi_agent_hermes_capacity
RESULTS=$RESEARCH/phase2_results
mkdir -p "$RESULTS"

RUNNER=$RESEARCH/phase2_workload_runner.py
BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
PORT=7109
LABEL=08_N3_ctx200K_textA
PARALLEL=3
PER_SLOT_CTX=204800
TOTAL_CTX=$(( PARALLEL * PER_SLOT_CTX ))
SLOTS="TL,Dev,QA"
DURATION=600
OUT=$RESULTS/${LABEL}__$(echo "$SLOTS" | tr ',' '_').json
LOG=$RESULTS/${LABEL}.bench.log

RAM_FLOOR_KB=$((13 * 1024 * 1024))
VRAM_FLOOR_MIB=400
WATCHDOG_RAM_KB=$((8 * 1024 * 1024))
WATCHDOG_VRAM_MIB=200

get_mem_avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

# Preflight
avail=$(get_mem_avail_kb)
if [ "$avail" -lt "$RAM_FLOOR_KB" ]; then
    echo "  !!! RAM PREFLIGHT FAIL: $((avail / 1024 / 1024)) GiB < $((RAM_FLOOR_KB / 1024 / 1024)) GiB"
    exit 1
fi
echo "  RAM preflight OK: $((avail / 1024 / 1024)) GiB"

echo "================================================================"
echo "=== $LABEL: --parallel $PARALLEL ctx_per_slot=$PER_SLOT_CTX (TEXT-ONLY)"
echo "================================================================"
echo "  booting llama-server (total -c $TOTAL_CTX, no mmproj)..."

GGML_CUDA_NO_PINNED=1 "$BIN" \
    --model "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --parallel "$PARALLEL" \
    --cont-batching \
    --ctx-size "$TOTAL_CTX" \
    --threads 16 \
    --n-gpu-layers -1 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn on \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":true}' \
    > "$LOG" 2>&1 &
LLAMA_PID=$!

# Wait /health
waited=0
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -gt 300 ]; then
        echo "  !!! TIMEOUT after ${waited}s"
        tail -15 "$LOG"
        kill $LLAMA_PID 2>/dev/null
        wait $LLAMA_PID 2>/dev/null
        exit 1
    fi
done
echo "  health OK after ${waited}s"

gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "  VRAM after boot: ${gpu_used} MiB used / ${gpu_free} MiB free"
echo "  MemAvailable after boot: $((avail / 1024 / 1024)) GiB"

if [ "$gpu_free" -lt "$VRAM_FLOOR_MIB" ]; then
    echo "  !!! VRAM PREFLIGHT FAIL"
    kill $LLAMA_PID 2>/dev/null
    wait $LLAMA_PID 2>/dev/null
    exit 1
fi

# Watchdog
PARENT=$$
(
    breach=0
    while true; do
        sleep 5
        a=$(get_mem_avail_kb)
        v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if [ "$a" -lt "$WATCHDOG_RAM_KB" ] || [ "$v" -lt "$WATCHDOG_VRAM_MIB" ]; then
            breach=$((breach + 5))
            echo "  !!! WATCHDOG: avail=$((a / 1024 / 1024)) GiB vfree=${v} MiB (${breach}s)"
            if [ "$breach" -ge 15 ]; then
                echo "  !!! WATCHDOG TRIGGERED"
                kill -TERM $PARENT 2>/dev/null
                exit 1
            fi
        else
            breach=0
        fi
    done
) &
WD=$!

echo "  running workload (${DURATION}s)..."
uv run python "$RUNNER" \
    --base-url "http://127.0.0.1:$PORT" \
    --slots "$SLOTS" \
    --duration "$DURATION" \
    --output "$OUT" \
    --config-label "$LABEL" 2>&1 | tail -40 || echo "  !!! workload errored"
echo "  workload complete; summary at $OUT"

kill $WD 2>/dev/null; wait $WD 2>/dev/null
echo "  stopping llama-server (pid $LLAMA_PID)..."
kill $LLAMA_PID
wait $LLAMA_PID 2>/dev/null
sleep 8

echo "=== Restoring hermes-agent ==="
uv run gpumod mode switch hermes-agent 2>&1 | tail -3
echo "=== Done. ==="
