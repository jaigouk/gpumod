#!/usr/bin/env bash
# gpumod-8viu Phase 2 v2 — slot save/restore smoke test, post-freeze redesign.
#
# v1 froze the host with --swa-full + ctx=393216 (3 × 128K). Diagnosis:
# --swa-full forces every layer to allocate full-ctx KV instead of the
# 1024-token sliding window, multiplying VRAM cost by ~30× for Gemma 4 and
# blowing past the 24 GiB ceiling. The cudaMalloc stall triggered the
# cudaHostAlloc-class freeze (gpumod-x7rv pattern) despite GGML_CUDA_NO_PINNED=1.
#
# v2 takes the cautious route:
#   - DO NOT use --swa-full. Test slot save/restore without it first. If
#     restored conversations behave correctly, we ship without the flag.
#   - Phase A: --parallel 1 --ctx-size 16384 (tiny KV, near-zero blast radius)
#   - Phase B: scale to --parallel 3 --ctx-size 393216 ONLY after A confirms
#     the API works.
#   - Hard pre-launch VRAM/RAM preflight before each phase.
#   - Watchdog: if VRAM free < 500 MiB sustained 10 s, kill.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260605_slot_persistence
mkdir -p "$RESEARCH"
RESULTS=$RESEARCH/phase2_smoke_v2_results.txt
: > "$RESULTS"

BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
SLOT_DIR=$HOME/.cache/gpumod-slot-test
PORT=7109

RAM_FLOOR_KB=$((13 * 1024 * 1024))
VRAM_FLOOR_MIB=500

mkdir -p "$SLOT_DIR"
rm -f "$SLOT_DIR"/*.bin

log() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

preflight() {
    local avail
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$avail" -lt "$RAM_FLOOR_KB" ]; then
        log "  !!! RAM preflight FAIL: $((avail / 1024 / 1024)) GiB"
        return 1
    fi
    local vfree
    vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    log "  preflight OK: RAM=$((avail / 1024 / 1024)) GiB, VRAM free=${vfree} MiB"
}

start_watchdog() {
    local parent=$$
    (
        local breach=0
        while true; do
            sleep 5
            local vfree
            vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
            if [ "$vfree" -lt "$VRAM_FLOOR_MIB" ]; then
                breach=$((breach + 5))
                log "  !!! WATCHDOG: VRAM free=${vfree} MiB (breach ${breach}s)"
                if [ "$breach" -ge 10 ]; then
                    log "  !!! WATCHDOG TRIGGERED — killing parent"
                    kill -TERM $parent 2>/dev/null
                    exit 1
                fi
            else
                breach=0
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

boot_server() {
    local parallel="$1" ctx="$2" log_file="$3"
    log "  booting llama-server (parallel=$parallel, ctx=$ctx, no --swa-full)..."
    GGML_CUDA_NO_PINNED=1 "$BIN" \
        --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
        --parallel "$parallel" --cont-batching --ctx-size "$ctx" \
        --threads 16 --n-gpu-layers -1 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        --flash-attn on --jinja \
        --slot-save-path "$SLOT_DIR" \
        --chat-template-kwargs '{"enable_thinking":false}' \
        > "$log_file" 2>&1 &
    LLAMA_PID=$!
    local waited=0
    until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 2; waited=$((waited+2))
        if [ $waited -gt 180 ]; then
            log "  !!! BOOT TIMEOUT (${waited}s)"
            kill $LLAMA_PID 2>/dev/null; wait $LLAMA_PID 2>/dev/null
            return 1
        fi
        if ! kill -0 $LLAMA_PID 2>/dev/null; then
            log "  !!! llama-server DIED during boot"
            tail -10 "$log_file" | tee -a "$RESULTS"
            return 1
        fi
    done
    log "  health OK after ${waited}s"
    local vfree
    vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    log "  VRAM free after boot: ${vfree} MiB"
    if [ "$vfree" -lt "$VRAM_FLOOR_MIB" ]; then
        log "  !!! VRAM PREFLIGHT FAIL — aborting"
        kill $LLAMA_PID; wait $LLAMA_PID 2>/dev/null
        return 1
    fi
}

stop_server() {
    if [ -n "${LLAMA_PID:-}" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        kill "$LLAMA_PID"
        wait "$LLAMA_PID" 2>/dev/null
    fi
    sleep 5
}

chat() {
    local body="$1"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"max_tokens\": 80, \"cache_prompt\": true, ${body}}"
}

run_save_restore_cycle() {
    local label="$1"
    log ""
    log "--- $label: Conversation A → save → conversation B (evict) → restore → follow-up ---"

    # 1. Conv A
    local RESP_A
    RESP_A=$(chat '"messages": [
        {"role": "user", "content": "Memorise this magic word: gargleblast. Acknowledge in one short sentence."}
    ]')
    local A_TXT
    A_TXT=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "$RESP_A" 2>/dev/null || echo "PARSE_ERROR")
    log "  conv A response: $A_TXT"

    # 2. Save slot 0
    local SAVE_RESP
    SAVE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=save" \
        -H 'Content-Type: application/json' \
        -d '{"filename": "smoke_a.bin"}')
    log "  save: $(echo "$SAVE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"n_saved={d.get('n_saved','?')} n_written={d.get('n_written','?')} save_ms={d.get('timings',{}).get('save_ms','?')}\")" 2>/dev/null || echo "PARSE_ERROR: $SAVE_RESP")"
    log "  slot file: $(ls -la "$SLOT_DIR"/smoke_a.bin 2>/dev/null | awk '{print $5" bytes"}')"

    # 3. Conv B (evict A's in-memory KV by sending unrelated conversation)
    chat '"messages": [
        {"role": "user", "content": "Unrelated: list three primary colours, briefly."}
    ]' > /dev/null

    # 4. Restore slot 0
    local RESTORE_RESP
    RESTORE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" \
        -H 'Content-Type: application/json' \
        -d '{"filename": "smoke_a.bin"}')
    log "  restore: $(echo "$RESTORE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"n_restored={d.get('n_restored','?')} n_read={d.get('n_read','?')} restore_ms={d.get('timings',{}).get('restore_ms','?')}\")" 2>/dev/null || echo "PARSE_ERROR: $RESTORE_RESP")"

    # 5. Follow-up after restore — does the model remember "gargleblast"?
    local RESP_A2
    RESP_A2=$(chat '"messages": [
        {"role": "user", "content": "Memorise this magic word: gargleblast. Acknowledge in one short sentence."},
        {"role": "assistant", "content": "Got it — the magic word is gargleblast."},
        {"role": "user", "content": "What was the magic word?"}
    ]')
    local A2_TXT
    A2_TXT=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "$RESP_A2" 2>/dev/null || echo "PARSE_ERROR")
    log "  follow-up response: $A2_TXT"

    if echo "$A2_TXT" | grep -qi "gargleblast"; then
        log "  ✓ $label PASS: model recalled the magic word from restored KV"
        return 0
    else
        log "  ✗ $label FAIL: model did NOT recall the magic word — restore may be incorrect for SWA layers without --swa-full"
        return 1
    fi
}

# ----------------------------------------------------------------------------
# Phase A: tiny scale, 1 slot, 16K ctx
# ----------------------------------------------------------------------------
log "=== Phase A: tiny scale (parallel=1, ctx=16384) ==="
preflight || exit 1
start_watchdog
if ! boot_server 1 16384 "$RESEARCH/phase2_smoke_v2_phaseA.bench.log"; then
    stop_watchdog; exit 1
fi
run_save_restore_cycle "Phase A" || PHASEA_FAIL=1
stop_server
stop_watchdog
log ""

if [ -n "${PHASEA_FAIL:-}" ]; then
    log "=== Phase A failed — STOPPING. Investigate before attempting production-scale Phase B. ==="
    exit 1
fi

# ----------------------------------------------------------------------------
# Phase B: production scale, 3 slots, 128K per slot
# ----------------------------------------------------------------------------
log "=== Phase B: production scale (parallel=3, ctx=393216 = 3 × 128K) ==="
preflight || exit 1
start_watchdog
if ! boot_server 3 393216 "$RESEARCH/phase2_smoke_v2_phaseB.bench.log"; then
    stop_watchdog; exit 1
fi
run_save_restore_cycle "Phase B" || PHASEB_FAIL=1
stop_server
stop_watchdog
log ""

log "=== Done. ==="
log "Results: $RESULTS"
log "Phase A: ${PHASEA_FAIL:+FAILED}${PHASEA_FAIL:-PASSED}"
log "Phase B: ${PHASEB_FAIL:+FAILED}${PHASEB_FAIL:-PASSED}"
