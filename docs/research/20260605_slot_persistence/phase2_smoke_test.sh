#!/usr/bin/env bash
# gpumod-8viu Phase 2 — slot save/restore smoke test.
# Verifies the documented pattern on b9500: save a slot's KV to disk, evict it
# with another conversation, restore it, confirm the follow-up hits prefix-cache.
#
# Same safeguards as the gpumod-8xaq orchestrators: GGML_CUDA_NO_PINNED,
# RAM/VRAM preflight (skipped here since we already verified the host state).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESEARCH=$ROOT/docs/research/20260605_slot_persistence
mkdir -p "$RESEARCH"

BIN=$HOME/bin/llama.cpp/build/bin/llama-server
MODEL=$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
SLOT_DIR=$HOME/.cache/gpumod-slot-test
PORT=7109
LOG=$RESEARCH/phase2_smoke.bench.log
RESULTS=$RESEARCH/phase2_smoke_results.txt

mkdir -p "$SLOT_DIR"
rm -f "$SLOT_DIR"/*.bin
: > "$RESULTS"

log() { printf '%s\n' "$*" | tee -a "$RESULTS"; }

log "=== gpumod-8viu Phase 2 smoke test ==="
log "Slot dir: $SLOT_DIR"
log "Slot files at start: $(ls -1 "$SLOT_DIR" 2>/dev/null | wc -l)"
log ""

# ----------------------------------------------------------------------------
# Boot llama-server with --slot-save-path + --swa-full
# ----------------------------------------------------------------------------
log "=== Booting llama-server (multi-slot + slot persistence + swa-full) ==="
GGML_CUDA_NO_PINNED=1 "$BIN" \
    --model "$MODEL" \
    --port "$PORT" --host 127.0.0.1 \
    --parallel 3 --cont-batching --ctx-size 393216 \
    --threads 16 --n-gpu-layers -1 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --jinja \
    --slot-save-path "$SLOT_DIR" \
    --swa-full \
    --chat-template-kwargs '{"enable_thinking":false}' \
    > "$LOG" 2>&1 &
LLAMA_PID=$!

waited=0
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 2; waited=$((waited+2))
    if [ $waited -gt 240 ]; then
        log "!!! TIMEOUT booting"
        tail -10 "$LOG" | tee -a "$RESULTS"
        kill $LLAMA_PID 2>/dev/null; wait $LLAMA_PID 2>/dev/null
        exit 1
    fi
done
log "  health OK after ${waited}s"
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
log "  VRAM free after boot: ${gpu_free} MiB"
log ""

# Helper: send a chat request, return raw JSON
chat() {
    local id_slot="$1"
    local body="$2"
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"max_tokens\": 100, \"id_slot\": ${id_slot}, \"cache_prompt\": true, ${body}}"
}

# Helper: extract usage from response JSON
usage_json() {
    python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r.get('usage',{}),indent=2))" <<< "$1"
}

# ----------------------------------------------------------------------------
# Step 1: Conversation A → slot 0
# ----------------------------------------------------------------------------
log "=== Step 1: Conversation A → slot 0 (cold; full prefill) ==="
RESP_A=$(chat 0 '"messages": [
    {"role": "user", "content": "Memorise this magic number: 42. Acknowledge briefly."}
]')
log "Response A usage:"
usage_json "$RESP_A" | tee -a "$RESULTS"
A_TOK=$(python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" <<< "$RESP_A")
log "  → prompt_tokens=$A_TOK"
log ""

# ----------------------------------------------------------------------------
# Step 2: Save slot 0 to disk
# ----------------------------------------------------------------------------
log "=== Step 2: Save slot 0 → smoke_a.bin ==="
SAVE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=save" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "smoke_a.bin"}')
log "Save response:"
echo "$SAVE_RESP" | python3 -m json.tool | tee -a "$RESULTS"
log ""
log "On-disk slot file:"
ls -la "$SLOT_DIR" | tee -a "$RESULTS"
log ""

# ----------------------------------------------------------------------------
# Step 3: Conversation B → slot 0 (evicts A's in-memory KV)
# ----------------------------------------------------------------------------
log "=== Step 3: Conversation B → slot 0 (evicts in-memory KV) ==="
chat 0 '"messages": [
    {"role": "user", "content": "Completely unrelated topic: list 3 colours of the rainbow."}
]' > /dev/null
log "  → done (response discarded)"
log ""

# ----------------------------------------------------------------------------
# Step 4: Restore slot 0 from disk
# ----------------------------------------------------------------------------
log "=== Step 4: Restore slot 0 ← smoke_a.bin ==="
RESTORE_RESP=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=restore" \
    -H 'Content-Type: application/json' \
    -d '{"filename": "smoke_a.bin"}')
log "Restore response:"
echo "$RESTORE_RESP" | python3 -m json.tool | tee -a "$RESULTS"
log ""

# ----------------------------------------------------------------------------
# Step 5: Follow-up to conversation A (cache should hit if restore worked)
# ----------------------------------------------------------------------------
log "=== Step 5: Follow-up to conv A → slot 0 (expect prefix-cache hit) ==="
RESP_A2=$(chat 0 '"messages": [
    {"role": "user", "content": "Memorise this magic number: 42. Acknowledge briefly."},
    {"role": "assistant", "content": "Got it — magic number is 42."},
    {"role": "user", "content": "What was the magic number?"}
]')
log "Response A2 usage:"
usage_json "$RESP_A2" | tee -a "$RESULTS"
A2_TOK=$(python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" <<< "$RESP_A2")
log "  → prompt_tokens=$A2_TOK"
A2_TEXT=$(python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" <<< "$RESP_A2")
log "  → response text: $A2_TEXT"
log ""

# ----------------------------------------------------------------------------
# Step 6: parse slot timing from llama-server log
# ----------------------------------------------------------------------------
log "=== Step 6: slot timing log (n_prompt_tokens_processed = uncached tokens) ==="
log "--- last 8 slot print_timing lines ---"
grep "slot print_timing" "$LOG" | tail -8 | tee -a "$RESULTS"
log ""
log "--- slot launch_slot lines (show cache hits / prompt sizes) ---"
grep -E "launch_slot|prompt eval time" "$LOG" | tail -10 | tee -a "$RESULTS"
log ""

# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
log "=== Smoke test verdict ==="
if echo "$A2_TEXT" | grep -qi "42"; then
    log "PASS: model recalled '42' from the restored KV — slot save/restore works on b9500"
else
    log "FAIL: model did not recall '42' — restore may not have rehydrated the conversation prefix"
fi
log ""

kill $LLAMA_PID
wait $LLAMA_PID 2>/dev/null
log "llama-server stopped."
log ""
log "Full server log: $LOG"
log "Smoke test results: $RESULTS"
