#!/usr/bin/env bash
# gpumod-8xaq Phase 1 — VRAM ceiling measurement for multi-slot llama-server
# configs. Co-tenant: vllm-embedding-code already running (2.5 GB).
# All measurements taken from /proc/<llama-server-pid>/used_gpu_memory via
# nvidia-smi --query-compute-apps. We isolate llama-server's bytes, not GPU total.
set -uo pipefail

BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
MMPROJ=$HOME/bin/gemma-4-26B-A4B-it-mmproj-BF16.gguf
PORT=7109
RESULTS=/tmp/phase1_results.tsv

echo -e "config\tN\tper_slot_ctx\ttotal_ctx\tllama_mb_after_load\tgpu_total_mb\tgpu_free_mb\tboot_secs" > "$RESULTS"

run_one() {
    local label="$1" parallel="$2" per_slot_ctx="$3"
    local total_ctx=$(( parallel * per_slot_ctx ))
    echo
    echo "=== $label: --parallel $parallel, per_slot_ctx=$per_slot_ctx, total -c $total_ctx ==="

    local t0=$(date +%s)
    "$BIN" \
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
        > /tmp/phase1_${label}.log 2>&1 &
    local pid=$!

    # Wait until /health responds OR timeout 240s
    local waited=0
    until curl -fsS http://127.0.0.1:$PORT/health >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -gt 240 ]; then
            echo "  !!! TIMEOUT after ${waited}s"
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            return 1
        fi
        # Check if process died (OOM, error)
        if ! kill -0 $pid 2>/dev/null; then
            echo "  !!! process died during boot (likely OOM)"
            tail -10 /tmp/phase1_${label}.log
            return 1
        fi
    done
    local t1=$(date +%s)
    local boot=$((t1 - t0))
    echo "  health OK after ${boot}s"

    # Measure: llama-server's GPU memory + GPU totals
    sleep 1  # let allocations settle
    local llama_mb=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$pid '$1==p {print $2}' | head -1)
    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    local gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    echo "  llama-server VRAM: ${llama_mb} MiB  |  GPU total used: ${gpu_used} MiB  |  GPU free: ${gpu_free} MiB"

    echo -e "${label}\t${parallel}\t${per_slot_ctx}\t${total_ctx}\t${llama_mb}\t${gpu_used}\t${gpu_free}\t${boot}" >> "$RESULTS"

    # Kill llama-server cleanly
    echo "  stopping llama-server (pid $pid)"
    kill $pid
    wait $pid 2>/dev/null
    sleep 5  # quiesce for driver to reclaim
}

# Configs from the spike plan
run_one  "N3_ctx128K"  3  131072
run_one  "N3_ctx64K"   3   65536
run_one  "N5_ctx64K"   5   65536
run_one  "N5_ctx32K"   5   32768
run_one  "N7_ctx32K"   7   32768

echo
echo "=== PHASE 1 RESULTS ==="
column -t -s $'\t' "$RESULTS"
