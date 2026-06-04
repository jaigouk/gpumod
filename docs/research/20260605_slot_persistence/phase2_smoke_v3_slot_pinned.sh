#!/usr/bin/env bash
# gpumod-8viu Phase 2 v3 — slot-pinned multi-slot save/restore.
#
# v2 Phase B saved 44 bytes (empty slot) because the chat request didn't
# pin to a specific slot. The conversation landed on whatever slot was idle
# (1 or 2), so save/restore on slot 0 was a no-op. The "PASS" was the
# prefix cache hiding the real result.
#
# v3 explicitly passes id_slot=0 (undocumented passthrough on
# /v1/chat/completions per ggml-org/llama.cpp#22354) so each step targets
# the same slot.
#
# Same safeguards: GGML_CUDA_NO_PINNED=1, no --swa-full, watchdog.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260605_slot_persistence
mkdir -p "$RESEARCH"
RESULTS=$RESEARCH/phase2_smoke_v3_results.txt
: > "$RESULTS"

BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
SLOT_DIR=$HOME/.cache/gpumod-slot-test
PORT=7109
VRAM_FLOOR_MIB=500

mkdir -p "$SLOT_DIR"
rm -f "$SLOT_DIR"/*.bin
log() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

# Watchdog
PARENT=$$
(
    breach=0
    while true; do
        sleep 5
        vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        if [ "$vfree" -lt "$VRAM_FLOOR_MIB" ]; then
            breach=$((breach+5))
            log "  !!! WATCHDOG: VRAM free=${vfree} MiB (breach ${breach}s)"
            if [ "$breach" -ge 10 ]; then
                log "  !!! WATCHDOG TRIGGERED"
                kill -TERM $PARENT 2>/dev/null
                exit 1
            fi
        else
            breach=0
        fi
    done
) & WD=$!

log "=== Phase 2 v3: slot-pinned multi-slot save/restore ==="
log ""

# Boot multi-slot
log "Booting llama-server (parallel=3, ctx=393216, --slot-save-path, NO --swa-full)..."
GGML_CUDA_NO_PINNED=1 "$BIN" \
    --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
    --parallel 3 --cont-batching --ctx-size 393216 \
    --threads 16 --n-gpu-layers -1 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --jinja \
    --slot-save-path "$SLOT_DIR" \
    --chat-template-kwargs '{"enable_thinking":false}' \
    > "$RESEARCH/phase2_smoke_v3.bench.log" 2>&1 &
LLAMA_PID=$!

waited=0
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 2; waited=$((waited+2))
    if [ $waited -gt 180 ]; then
        log "!!! BOOT TIMEOUT"
        kill $LLAMA_PID 2>/dev/null; wait $LLAMA_PID 2>/dev/null
        kill $WD 2>/dev/null
        exit 1
    fi
done
log "  health OK after ${waited}s"
log "  VRAM free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') MiB"

# Helper: chat with id_slot=N pinning
chat_slot() {
    local id_slot="$1" body="$2"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"max_tokens\": 80, \"cache_prompt\": true, \"id_slot\": ${id_slot}, ${body}}"
}

# 1. Conv A → slot 0 (pinned)
log ""
log "Step 1: Conv A → slot 0 (id_slot=0)"
RESP_A=$(chat_slot 0 '"messages": [
    {"role": "user", "content": "Memorise this magic word: gargleblast. Acknowledge in one short sentence."}
]')
A_TXT=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "$RESP_A" 2>/dev/null || echo "PARSE_ERROR")
log "  → response: $A_TXT"

# 2. Save slot 0
log ""
log "Step 2: Save slot 0"
SAVE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=save" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "smoke_v3_slot0.bin"}')
log "  raw: $SAVE_RESP"
SAVE_PARSED=$(python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"n_saved={d.get('n_saved','?')} n_written={d.get('n_written','?')} save_ms={d.get('timings',{}).get('save_ms','?')}\")" <<< "$SAVE_RESP" 2>/dev/null || echo "PARSE_ERROR")
log "  → $SAVE_PARSED"
log "  slot file: $(ls -la "$SLOT_DIR"/smoke_v3_slot0.bin 2>/dev/null | awk '{print $5}') bytes"

# 3. Conv B → slot 0 (pinned, evicts A)
log ""
log "Step 3: Conv B → slot 0 (id_slot=0) — evicts A's in-memory KV"
chat_slot 0 '"messages": [
    {"role": "user", "content": "Totally unrelated topic — list five animals briefly."}
]' > /dev/null
log "  → done"

# 4. Restore slot 0
log ""
log "Step 4: Restore slot 0"
RESTORE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "smoke_v3_slot0.bin"}')
log "  raw: $RESTORE_RESP"
RESTORE_PARSED=$(python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"n_restored={d.get('n_restored','?')} n_read={d.get('n_read','?')} restore_ms={d.get('timings',{}).get('restore_ms','?')}\")" <<< "$RESTORE_RESP" 2>/dev/null || echo "PARSE_ERROR")
log "  → $RESTORE_PARSED"

# 5. Follow-up to A → slot 0 (use restored KV)
log ""
log "Step 5: Follow-up to A → slot 0"
RESP_A2=$(chat_slot 0 '"messages": [
    {"role": "user", "content": "Memorise this magic word: gargleblast. Acknowledge in one short sentence."},
    {"role": "assistant", "content": "Got it — the magic word is gargleblast."},
    {"role": "user", "content": "What was the magic word?"}
]')
A2_TXT=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "$RESP_A2" 2>/dev/null || echo "PARSE_ERROR")
log "  → response: $A2_TXT"

# 6. Server log tail showing slot eval counts
log ""
log "Step 6: slot timing log tail (look for n_prompt_tokens_processed)"
grep -E "slot launch_slot_|slot release|slot print_timing|prompt eval time" "$RESEARCH/phase2_smoke_v3.bench.log" | tail -15 | tee -a "$RESULTS"

# Verdict
log ""
log "=== Verdict ==="
N_SAVED=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('n_saved',0))" <<< "$SAVE_RESP" 2>/dev/null || echo "0")
if [ "$N_SAVED" -gt 0 ] && echo "$A2_TXT" | grep -qi "gargleblast"; then
    log "PASS: multi-slot save/restore works correctly with id_slot pinning"
    log "  - saved $N_SAVED tokens of KV to disk"
    log "  - restored & follow-up correctly recalled the magic word"
else
    log "FAIL: n_saved=$N_SAVED, follow-up=$A2_TXT"
fi

kill $LLAMA_PID; wait $LLAMA_PID 2>/dev/null
kill $WD 2>/dev/null; wait $WD 2>/dev/null
log ""
log "=== Done. ==="
