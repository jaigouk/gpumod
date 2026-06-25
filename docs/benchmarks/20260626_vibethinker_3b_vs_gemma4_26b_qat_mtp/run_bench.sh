#!/usr/bin/env bash
# gpumod-msy8 benchmark driver: VibeThinker-3B Q8_0 vs Gemma 4 26B-A4B QAT-MTP.
#
# Runs the VibeThinker arm ONLY. The Gemma 4 26B-A4B QAT-MTP baseline is REUSED
# from docs/benchmarks/20260625_unified/result_gemma4-26b-a4b-qat-mtp-q4.json
# (already copied into this folder): same llama.cpp binary b9784, same v2 coding
# suite L1-L5, same GEMMA_CODING sampler, same 15 iterations, dated 2026-06-25
# (one day prior). Quality (score/pass-rate) is host-load-insensitive and fully
# comparable; the README flags the 1-day host-load window for TPS interpretation.
#
# VibeThinker runs on port 7115 — NOT the daily-driver port 7110 — so no
# hermes-agent orchestrator can reconnect to the model-under-test mid-run.
#
# Launch inside a tmux 'bench' session and pair with a separate 'monitor' session
# per .claude/CLAUDE.md "Running Long Benchmarks":
#   tmux new -s bench
#   ./docs/benchmarks/20260626_vibethinker_3b_vs_gemma4_26b_qat_mtp/run_bench.sh
#   (detach Ctrl-b d)
# Monitor (separate session, not a pane):
#   tmux new -s monitor   # then split: nvidia-smi -l 5 / journalctl --user -u
#   vibethinker-3b-q8.service -f / watch -n5 'free -h && dmesg | tail -3'
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="docs/benchmarks/20260626_vibethinker_3b_vs_gemma4_26b_qat_mtp"
MODEL="vibethinker-3b-q8"
PORT=7115

# VRAM isolation per .claude/CLAUDE.md: stop all gpumod-tracked services before
# the model start. Idempotent — a no-op if mode is already blank.
echo "=== $(date -Iseconds) ensuring blank mode (VRAM isolation) ==="
uv run gpumod mode switch blank

# Quiesce window: a stop here is heavy-GPU-eligible and trips the start-side
# quiesce gate on the model-under-test. 15s is conservative (lesson: gpumod-rjkx
# first-launch failure).
echo "=== $(date -Iseconds) quiesce wait (15s) ==="
sleep 15

echo "=== $(date -Iseconds) starting service $MODEL on port $PORT ==="
if ! uv run gpumod service start "$MODEL"; then
    echo "!!! service start failed for $MODEL — aborting"
    exit 1
fi

echo "=== waiting for /health on $PORT ==="
waited=0
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -gt 300 ]; then
        echo "!!! $MODEL failed to come up in 5min — aborting"
        uv run gpumod service stop "$MODEL" || true
        exit 1
    fi
done
echo "=== health OK after ${waited}s, launching benchmark ==="

uv run python scripts/run_qwen36_benchmark.py \
    --model "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee -a "$OUT/run_${MODEL}.log"

echo "=== $(date -Iseconds) benchmark $MODEL done, stopping service ==="
uv run gpumod service stop "$MODEL"

echo "=== $(date -Iseconds) VIBETHINKER BENCHMARK COMPLETE ==="
echo "Gemma baseline reused (no run): $OUT/result_gemma4-26b-a4b-qat-mtp-q4.json"
