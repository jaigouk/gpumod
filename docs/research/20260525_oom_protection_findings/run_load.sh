#!/usr/bin/env bash
# Drive a single instrumented load of qwen36-35b-a3b-mtp-iq4xs-preserve.
# Captures the PID of the systemd unit's main process, attaches sampler.py,
# polls /health, and saves a tidy CSV per run.
#
# Usage:
#   ./run_load.sh <run-label>
#
# Pre-requisites:
#   - Service is configured but stopped
#   - MemAvailable >= 20125 MB (run drop_caches first if not)
#   - Working directory is the repo root

set -euo pipefail

LABEL="${1:?usage: $0 <run-label>}"
SERVICE="qwen36-35b-a3b-mtp-iq4xs-preserve"
PORT=7104
OUTDIR="docs/research/20260525_oom_protection_findings/runs/${LABEL}"

mkdir -p "${OUTDIR}"
echo "[$(date -Iseconds)] === run_load.sh label=${LABEL} service=${SERVICE} ==="

# Snapshot pre-load state
{
    echo "## Pre-load"
    echo "ts=$(date -Iseconds)"
    grep -E "MemTotal|MemFree|MemAvailable|Cached|SwapFree|SwapTotal|Pinned|Mlocked" /proc/meminfo
    echo "psi:"; cat /proc/pressure/memory
    echo "nvidia-smi:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo "swap:"; cat /proc/swaps
    echo "cgroup memory.current ($SERVICE):"
    cat "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/${SERVICE}.service/memory.current" 2>/dev/null || echo "(service not running)"
} > "${OUTDIR}/pre.txt"

# Kick off the service via systemctl (non-blocking, so we can attach sampler
# during the load instead of after /health is up)
START=$(date +%s.%N)
echo "[$(date -Iseconds)] starting service via systemctl --no-block..."
systemctl --user start --no-block "${SERVICE}.service" 2>&1 | tee "${OUTDIR}/start.log"

# Get the main PID (poll fast — load starts within ~1s of systemctl returning)
PID=""
for i in $(seq 1 200); do
    PID=$(systemctl --user show -p MainPID --value "${SERVICE}.service" 2>/dev/null || echo "0")
    [ "${PID:-0}" != "0" ] && break
    # Also surface ExecStartPre failures early (preflight blocked the start)
    STATE=$(systemctl --user show -p SubState --value "${SERVICE}.service" 2>/dev/null || echo "")
    if [ "${STATE}" = "failed" ]; then
        echo "[$(date -Iseconds)] ERROR: ${SERVICE} entered 'failed' state during start (preflight or ExecStart)" >&2
        journalctl --user -u "${SERVICE}.service" -n 30 --no-pager > "${OUTDIR}/journal_failed.log" || true
        exit 1
    fi
    sleep 0.05
done
if [ "${PID:-0}" = "0" ]; then
    echo "[$(date -Iseconds)] ERROR: could not resolve MainPID for ${SERVICE}.service" >&2
    journalctl --user -u "${SERVICE}.service" -n 30 --no-pager > "${OUTDIR}/journal_failed.log" || true
    exit 1
fi
echo "[$(date -Iseconds)] MainPID=${PID}; attaching sampler"

# Start sampler in background (10 Hz)
SAMPLER_PID=""
python3 docs/research/20260525_oom_protection_findings/sampler.py \
    --pid "${PID}" \
    --out "${OUTDIR}/sampler.csv" \
    --duration 600 \
    --interval 0.1 &
SAMPLER_PID=$!
echo "[$(date -Iseconds)] sampler PID=${SAMPLER_PID}"

# Poll /health
HEALTH_AT=""
for i in $(seq 1 120); do
    if curl -sf -o /dev/null --max-time 2 "http://localhost:${PORT}/health" 2>/dev/null; then
        HEALTH_AT=$(date +%s.%N)
        echo "[$(date -Iseconds)] /health=200 after ${i} polls"
        break
    fi
    sleep 1
done

# Let it idle for 5s to capture steady state then stop sampler
sleep 5
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

# Calculate timings
if [ -n "${HEALTH_AT}" ]; then
    LOAD_SEC=$(python3 -c "print(f'{${HEALTH_AT} - ${START}:.2f}')")
else
    LOAD_SEC="FAILED"
fi

# Snapshot peak state
{
    echo "## Post-load (steady-state)"
    echo "ts=$(date -Iseconds)"
    echo "load_seconds=${LOAD_SEC}"
    grep -E "MemTotal|MemFree|MemAvailable|Cached|SwapFree|SwapTotal|Pinned|Mlocked" /proc/meminfo
    echo "psi:"; cat /proc/pressure/memory
    echo "nvidia-smi:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo "vmstat sample:"; cat "/proc/${PID}/status" 2>/dev/null | grep -E "VmRSS|VmPin|VmSwap|VmLck|VmData" || echo "(PID gone)"
    echo "cgroup memory.current ($SERVICE):"
    cat "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/${SERVICE}.service/memory.current" 2>/dev/null
    echo "cgroup memory.peak ($SERVICE):"
    cat "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/${SERVICE}.service/memory.peak" 2>/dev/null
} > "${OUTDIR}/post.txt"

# Stop service
echo "[$(date -Iseconds)] stopping service..."
systemctl --user stop "${SERVICE}.service" 2>&1 | tee -a "${OUTDIR}/start.log"

# Summary
echo "[$(date -Iseconds)] === DONE label=${LABEL} load_seconds=${LOAD_SEC} ==="
echo "  sampler csv: ${OUTDIR}/sampler.csv ($(wc -l < ${OUTDIR}/sampler.csv) rows)"
echo "  pre/post snapshots: ${OUTDIR}/pre.txt ${OUTDIR}/post.txt"
