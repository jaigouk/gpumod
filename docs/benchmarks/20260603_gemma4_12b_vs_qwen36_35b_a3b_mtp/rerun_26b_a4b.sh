#!/usr/bin/env bash
# gpumod-h6gs reproducibility check: re-run gemma4-26b-a4b-q4 (n=15) after
# archiving the original run as *.run1.*. Mirrors rerun_q8.sh for the MoE
# 26B-A4B variant, which outperformed gemma4-12b-q8 in run 1 — this rerun
# confirms whether that advantage is intrinsic to the architecture or
# sample variance from a single n=15 draw.
#
# Launch inside a tmux session and pair with a 'monitor' session per
# .claude/CLAUDE.md "Running Long Benchmarks". ~80-100 min wall-clock
# (slower per token than dense 12B Q8 due to ~4B active params + MoE
# routing overhead; faster than 35B-A3B class but slower than dense 12B).
set -euo pipefail
cd "$(dirname "$0")/../../.."

OUT="docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp"
MODEL="gemma4-26b-a4b-q4"
PORT=7109
QUIESCE_SECS=20

# Refuse to run if the service is currently active — we want a fresh load.
state=$(systemctl --user is-active "${MODEL}.service" 2>&1 || true)
if [ "$state" = "active" ]; then
    echo "!!! ${MODEL}.service is currently active. Stop it first:"
    echo "    uv run gpumod service stop ${MODEL}"
    exit 1
fi

# 26B-A4B file size guard: expect ~12.7 GiB (13597175584 bytes per HF manifest).
# Refuse to start if under 12 GiB — likely still downloading or truncated.
MODEL_FILE="$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "!!! $MODEL_FILE not found"
    exit 1
fi
SIZE_BYTES=$(stat -c%s "$MODEL_FILE")
MIN_BYTES=$((12 * 1024 * 1024 * 1024))
if [ "$SIZE_BYTES" -lt "$MIN_BYTES" ]; then
    echo "!!! $MODEL_FILE is only $((SIZE_BYTES / 1024 / 1024)) MiB — expected 12+ GiB. Download likely still in progress."
    exit 1
fi

# Archive prior "current" data to the next available run slot (run1, run2, ...).
# Idempotent: if no current data exists (e.g. an earlier attempt aborted post-archive),
# we just skip ahead to launching a fresh run.
if [ -f "$OUT/result_${MODEL}.json" ]; then
    n=1
    while [ -e "$OUT/result_${MODEL}.run${n}.json" ] \
       || [ -e "$OUT/run_${MODEL}.run${n}.log" ] \
       || [ -e "$OUT/artifacts/${MODEL}.run${n}" ]; do
        n=$((n + 1))
    done
    echo "=== $(date -Iseconds) archiving current data → run${n} ==="
    mv "$OUT/result_${MODEL}.json" "$OUT/result_${MODEL}.run${n}.json"
    [ -f "$OUT/run_${MODEL}.log" ]        && mv "$OUT/run_${MODEL}.log"   "$OUT/run_${MODEL}.run${n}.log"
    [ -d "$OUT/artifacts/${MODEL}" ]      && mv "$OUT/artifacts/${MODEL}" "$OUT/artifacts/${MODEL}.run${n}"
else
    echo "=== no current ${MODEL} result to archive — proceeding to fresh run ==="
fi

# VRAM isolation per .claude/CLAUDE.md "Running Long Benchmarks": stop all
# gpumod-tracked services before starting the model under test. Idempotent.
echo "=== $(date -Iseconds) ensuring blank mode (VRAM isolation) ==="
uv run gpumod mode switch blank

echo "=== $(date -Iseconds) starting service ${MODEL} on port ${PORT} ==="
uv run gpumod service start "$MODEL"

echo "=== waiting for /health on ${PORT} ==="
waited=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -gt 300 ]; then
        echo "!!! ${MODEL} failed to come up in 5min — aborting"
        uv run gpumod service stop "$MODEL" || true
        exit 1
    fi
done
echo "=== health OK after ${waited}s, launching benchmark ==="

uv run python scripts/run_qwen36_benchmark.py \
    --model "$MODEL" \
    --output-dir "$OUT" \
    2>&1 | tee -a "$OUT/run_${MODEL}.log"

echo "=== $(date -Iseconds) benchmark ${MODEL} done, stopping service ==="
uv run gpumod service stop "$MODEL"
echo "=== quiescing ${QUIESCE_SECS}s ==="
sleep "$QUIESCE_SECS"

echo "=== $(date -Iseconds) RE-RUN COMPLETE ==="
echo
echo "Compare with run 1:"
echo "  python3 -c \"import json; r1=json.load(open('$OUT/result_${MODEL}.run1.json'))['summary']; r2=json.load(open('$OUT/result_${MODEL}.json'))['summary']; print('run1:', r1['stats'], r1['scores']); print('run2:', r2['stats'], r2['scores'])\""
