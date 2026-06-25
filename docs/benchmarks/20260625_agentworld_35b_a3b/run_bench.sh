#!/usr/bin/env bash
# gpumod-qsgl.3 benchmark driver — AgentWorld-35B-A3B vs MoE peers (v2 coding suite).
#
# Runs all 4 arms on the SAME llama.cpp binary (b9784 / 8be759e6f) for a clean
# comparison (FIX-3: the prior gemma4 result JSON was on an older binary; we re-run).
# Each arm is started via its PRESET (gpumod service start) so AgentWorld's
# --override-kv conversion-defect fix applies through systemd — never a bare
# llama-server.  See docs/research/20260624_agentworld_qwen35moe_mtp/FINDINGS.md.
#
# CLAUDE.md long-benchmark protocol:
#   - run THIS inside a tmux session (e.g. `tmux new -s bench`)
#   - open a SEPARATE tmux session for monitoring (nvidia-smi / journalctl / free)
#   - VRAM isolation: this script switches to `blank` first; nothing else GPU-resident.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
OUT="docs/benchmarks/20260625_agentworld_35b_a3b"
ITERS="${ITERS:-15}"
QUIESCE="${QUIESCE:-15}"   # quiesce wait after each stop (hard-won fix, 20260606)
HEALTH_TIMEOUT=300
mkdir -p "$OUT"

# arm = "MODELS_key  service_id  port"  (MODELS key feeds --model; service_id starts the preset)
ARMS=(
  "agentworld-35b-a3b-q4    agentworld-35b-a3b-q4     7111"
  "qwen36-35b-a3b           qwen36-35b-a3b-q4         7101"
  "qwen36-35b-a3b-mtp-iq4xs qwen36-35b-a3b-mtp-iq4xs  7103"
  "gemma4-26b-a4b-qat-q4    gemma4-26b-a4b-qat-q4     7110"
)

echo "### VRAM isolation: mode switch blank ($(date -Is))"
uv run gpumod mode switch blank
sleep "$QUIESCE"

for arm in "${ARMS[@]}"; do
  read -r MKEY SVC PORT <<<"$arm"
  echo "==================== ARM: $MKEY  (svc=$SVC port=$PORT)  $(date -Is) ===================="
  if ! uv run gpumod service start "$SVC"; then
    echo "!!! service start failed for $SVC — aborting run"; exit 1
  fi
  echo "=== waiting for /health on $PORT (max ${HEALTH_TIMEOUT}s) ==="
  waited=0
  until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 3; waited=$((waited + 3))
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      echo "!!! /health timeout for $SVC after ${waited}s — aborting"
      uv run gpumod service stop "$SVC" || true
      exit 1
    fi
  done
  echo "=== health OK after ${waited}s — running $ITERS iters ==="
  uv run python scripts/run_qwen36_benchmark.py \
    --model "$MKEY" --iterations "$ITERS" --output-dir "$OUT" \
    2>&1 | tee "$OUT/run_${MKEY}.log"
  echo "=== stopping $SVC ==="
  uv run gpumod service stop "$SVC"
  sleep "$QUIESCE"
done

echo "==================== ALL ARMS DONE $(date -Is) ===================="
ls -1 "$OUT"/result_*.json
echo "GPU left in 'blank' — restore your daily driver with: uv run gpumod mode switch hermes-agent"
