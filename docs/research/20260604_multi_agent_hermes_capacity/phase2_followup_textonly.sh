#!/usr/bin/env bash
# gpumod-8xaq Phase 2 FOLLOW-UP — text-only configs. Drops --mmproj from
# llama-server boot to free ~1.1 GiB VRAM, enabling configs that were
# vision-blocked: N=3 ctx=128K, N=5 ctx=64K, N=3 ctx=256K, N=4 ctx=256K.
#
# Boots with the same safeguards as the main orchestrator:
#   GGML_CUDA_NO_PINNED=1, RAM preflight (≥13 GiB), VRAM preflight (≥400 MiB),
#   watchdog (kill on MemAvail<8 GiB or VRAM<200 MiB sustained 15s).
#
# Configs ordered by priority: the 256K runs land first because the user
# specifically asked about them (gpumod-8xaq follow-up).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260604_multi_agent_hermes_capacity
RESULTS=$RESEARCH/phase2_results
mkdir -p "$RESULTS"

RUNNER=$RESEARCH/phase2_workload_runner.py
BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
PORT=7109
DURATION=600  # 10 min per config

# Safeguard thresholds (same as main orchestrator)
RAM_FLOOR_KB=$((13 * 1024 * 1024))
VRAM_FLOOR_MIB=400
WATCHDOG_RAM_KB=$((8 * 1024 * 1024))
WATCHDOG_VRAM_MIB=200

get_mem_avail_kb() {
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
}

ram_preflight() {
    local avail
    avail=$(get_mem_avail_kb)
    if [ "$avail" -lt "$RAM_FLOOR_KB" ]; then
        local avail_gib=$((avail / 1024 / 1024))
        local floor_gib=$((RAM_FLOOR_KB / 1024 / 1024))
        echo "  !!! RAM PREFLIGHT FAIL: MemAvailable=${avail_gib} GiB < floor ${floor_gib} GiB"
        return 1
    fi
    echo "  RAM preflight OK: MemAvailable=$((avail / 1024 / 1024)) GiB"
}

start_watchdog() {
    local parent=$$
    (
        local breach_secs=0
        while true; do
            sleep 5
            local avail vfree
            avail=$(get_mem_avail_kb)
            vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
            if [ "$avail" -lt "$WATCHDOG_RAM_KB" ] || [ "$vfree" -lt "$WATCHDOG_VRAM_MIB" ]; then
                breach_secs=$((breach_secs + 5))
                echo "  !!! WATCHDOG: avail=$((avail / 1024 / 1024)) GiB vfree=${vfree} MiB (breach for ${breach_secs}s)"
                if [ "$breach_secs" -ge 15 ]; then
                    echo "  !!! WATCHDOG TRIGGERED — sustained breach, killing orchestrator"
                    kill -TERM $parent 2>/dev/null
                    exit 1
                fi
            else
                breach_secs=0
            fi
        done
    ) &
    WATCHDOG_PID=$!
}

stop_watchdog() {
    if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null
        wait "$WATCHDOG_PID" 2>/dev/null
    fi
}

wait_health() {
    local waited=0
    until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -gt 300 ]; then
            echo "  !!! TIMEOUT waiting for /health (${waited}s)"
            return 1
        fi
    done
    echo "  health OK after ${waited}s"
}

# Args: config_label parallel per_slot_ctx workload_slots
run_config() {
    local label="$1" parallel="$2" per_slot_ctx="$3" slots="$4"
    local total_ctx=$(( parallel * per_slot_ctx ))
    local out=$RESULTS/${label}__$(echo "$slots" | tr ',' '_' | tr -d ' ').json
    local log=$RESULTS/${label}.bench.log

    echo
    echo "================================================================"
    echo "=== $label: --parallel $parallel ctx_per_slot=$per_slot_ctx slots=$slots (TEXT-ONLY)"
    echo "================================================================"

    if ! ram_preflight; then
        echo "  SKIPPING $label — host RAM unsafe"
        return 1
    fi

    # NOTE: no --mmproj flag → text-only, saves ~1.1 GiB VRAM
    echo "  booting llama-server (total -c $total_ctx, no mmproj) with GGML_CUDA_NO_PINNED=1..."
    GGML_CUDA_NO_PINNED=1 "$BIN" \
        --model "$MODEL" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --parallel "$parallel" \
        --cont-batching \
        --ctx-size "$total_ctx" \
        --threads 16 \
        --n-gpu-layers -1 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --flash-attn on \
        --jinja \
        --chat-template-kwargs '{"enable_thinking":true}' \
        > "$log" 2>&1 &
    local pid=$!

    if ! wait_health; then
        echo "  !!! BOOT FAILED, tail of log:"
        tail -15 "$log"
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        return 1
    fi

    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    local gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    local mem_avail=$(get_mem_avail_kb)
    echo "  VRAM after boot: ${gpu_used} MiB used / ${gpu_free} MiB free"
    echo "  MemAvailable after boot: $((mem_avail / 1024 / 1024)) GiB"

    if [ "$gpu_free" -lt "$VRAM_FLOOR_MIB" ]; then
        echo "  !!! VRAM PREFLIGHT FAIL: ${gpu_free} MiB < floor ${VRAM_FLOOR_MIB} MiB"
        echo "  SKIPPING workload for $label (config does not fit safely)"
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        sleep 8
        return 1
    fi

    start_watchdog
    echo "  running workload (${DURATION}s)..."
    if uv run python "$RUNNER" \
        --base-url "http://127.0.0.1:$PORT" \
        --slots "$slots" \
        --duration "$DURATION" \
        --output "$out" \
        --config-label "$label" 2>&1 | tail -40; then
        echo "  workload complete; summary at $out"
    else
        echo "  !!! workload errored"
    fi
    stop_watchdog

    echo "  stopping llama-server (pid $pid)..."
    kill $pid
    wait $pid 2>/dev/null
    sleep 8
}

# ---------------------------------------------------------------------------
# Configs — 256K configs first (user priority), then 128K/64K text-only.
# Caller MUST stop the gpumod gemma service before invoking this script.
# vllm-embedding-code stays running (realistic Hermes co-tenant).
# ---------------------------------------------------------------------------

# User explicit asks: 256K configs first, then 200K sweet spot
run_config "07_N3_ctx256K_textA" 3  262144  "TL,Dev,QA"
run_config "08_N3_ctx200K_textA" 3  204800  "TL,Dev,QA"

echo
echo "=== Phase 2 follow-up complete. Restoring hermes-agent... ==="
sleep 5
uv run gpumod mode switch hermes-agent 2>&1 | tail -3

echo
echo "=== All Phase 2 result JSONs ==="
ls -la "$RESULTS"/*.json 2>&1 | tail -10
