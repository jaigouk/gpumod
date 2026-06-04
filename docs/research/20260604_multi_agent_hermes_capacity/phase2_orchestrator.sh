#!/usr/bin/env bash
# gpumod-8xaq Phase 2 orchestrator — boots llama-server at each config,
# runs the workload, captures the JSON summary, moves on. Keeps
# vllm-embedding-code running throughout for realistic Hermes co-tenancy.
# Re-starts hermes-agent at the end to restore production state.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260604_multi_agent_hermes_capacity
RESULTS=$RESEARCH/phase2_results
mkdir -p "$RESULTS"

RUNNER=$RESEARCH/phase2_workload_runner.py
BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
MMPROJ=$HOME/bin/gemma-4-26B-A4B-it-mmproj-BF16.gguf
PORT=7109
DURATION=600  # 10 min per config

# OOM safeguard thresholds.
# RAM_FLOOR_KB: refuse to boot if MemAvailable below this. 13 GiB is the
#   marginal floor from gpumod-x7rv (12 GiB = unrecoverable hang on this host).
# VRAM_FLOOR_MIB: refuse to launch workload if free VRAM below this after boot.
# WATCHDOG_RAM_KB: kill orchestrator if MemAvailable falls below this mid-run.
# WATCHDOG_VRAM_MIB: kill orchestrator if free VRAM falls below this sustained.
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
        echo "  !!! refusing to boot — cudaHostAlloc hang risk (gpumod-x7rv)"
        return 1
    fi
    echo "  RAM preflight OK: MemAvailable=$((avail / 1024 / 1024)) GiB"
}

# Side watchdog: SIGTERMs the orchestrator if RAM/VRAM crosses panic floors.
# Runs only while a workload is in flight; killed before each config tears down.
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

# Returns 0 if /health responds, nonzero otherwise
wait_health() {
    local waited=0
    until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -gt 240 ]; then
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
    echo "=== $label: --parallel $parallel ctx_per_slot=$per_slot_ctx slots=$slots"
    echo "================================================================"

    # OOM preflight
    if ! ram_preflight; then
        echo "  SKIPPING $label — host RAM unsafe"
        return 1
    fi

    # Boot llama-server in background.
    # GGML_CUDA_NO_PINNED=1 disables cudaMallocHost (the escape hatch from
    # gpumod-56md). Eliminates the cudaHostAlloc-class driver freeze when
    # MemAvailable is in the marginal 12-18 GiB range. Cost: ~0.3% TPS.
    echo "  booting llama-server (total -c $total_ctx) with GGML_CUDA_NO_PINNED=1..."
    GGML_CUDA_NO_PINNED=1 "$BIN" \
        --model "$MODEL" \
        --mmproj "$MMPROJ" \
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

    # Capture VRAM snapshot after boot
    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    local gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    local mem_avail=$(get_mem_avail_kb)
    echo "  VRAM after boot: ${gpu_used} MiB used / ${gpu_free} MiB free"
    echo "  MemAvailable after boot: $((mem_avail / 1024 / 1024)) GiB"

    if [ "$gpu_free" -lt "$VRAM_FLOOR_MIB" ]; then
        echo "  !!! VRAM PREFLIGHT FAIL: ${gpu_free} MiB < floor ${VRAM_FLOOR_MIB} MiB"
        echo "  SKIPPING workload for $label"
        kill $pid 2>/dev/null
        wait $pid 2>/dev/null
        sleep 8
        return 1
    fi

    # Run the workload under watchdog protection
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

    # Kill llama-server cleanly
    echo "  stopping llama-server (pid $pid)..."
    kill $pid
    wait $pid 2>/dev/null
    sleep 8  # quiesce for the GPU driver
}

# ---------------------------------------------------------------------------
# Stop the gpumod-managed gemma service to free VRAM. Keep
# vllm-embedding-code running.
# ---------------------------------------------------------------------------
echo "=== Stop gemma4-26b-a4b-q4 (gpumod service) for Phase 2 ==="
uv run gpumod service stop gemma4-26b-a4b-q4 2>&1 | tail -2
sleep 10

# ---------------------------------------------------------------------------
# The 6 test configs
# ---------------------------------------------------------------------------

# Baseline: single-slot 128K — apples-to-apples with the gpumod-h6gs bench
# numbers, with the new tool-overhead workload pattern applied.
run_config "01_N1_ctx128K"  1  131072  "Dev"

# Workflow A recommended: 3 asymmetric roles at 64K each
run_config "02_N3_ctx64K_A" 3   65536  "TL,Dev,QA"

# Workflow B same config: 3 uniform Research at 64K each
run_config "03_N3_ctx64K_B" 3   65536  "3xResearch"

# 5-slot Workflow A stretched: TL + 2 Dev + 2 QA at 32K each
run_config "04_N5_ctx32K_A" 5   32768  "TL,2xDev,2xQA"

# 5-slot mixed: Research-heavy team at 32K each
run_config "05_N5_ctx32K_B" 5   32768  "3xResearch,1xDev,1xTL"

# Tool storm: rapid tool-call churn, 3 slots at 64K each
run_config "06_N3_ctx64K_storm" 3 65536 "3xToolStorm"

# ---------------------------------------------------------------------------
# Restore production state
# ---------------------------------------------------------------------------
echo
echo "=== Phase 2 complete. Restoring hermes-agent... ==="
sleep 5
uv run gpumod mode switch hermes-agent 2>&1 | tail -3

echo
echo "=== All Phase 2 result JSONs ==="
ls -la "$RESULTS"/*.json 2>&1 | tail -10
