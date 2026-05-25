#!/usr/bin/env bash
# Run llama-server directly (bypassing systemd preflight) with sampler attached.
# Used for Phase 4 controlled-pressure tests.
#
# Usage:
#   ./run_load_direct.sh <run-label>
#
# Reads the rendered unit file to extract the ExecStart command (so it stays
# in sync with the preset). Stops on /health=200 or after 120s timeout.

set -euo pipefail

LABEL="${1:?usage: $0 <run-label>}"
SERVICE="qwen36-35b-a3b-mtp-iq4xs-preserve"
PORT=7104
UNIT="$HOME/.config/systemd/user/${SERVICE}.service"
OUTDIR="docs/research/20260525_oom_protection_findings/runs/${LABEL}"

mkdir -p "${OUTDIR}"
echo "[$(date -Iseconds)] === run_load_direct.sh label=${LABEL} ==="

# Defensive: kill any stale llama-server bound to our port from prior runs
if STALE=$(lsof -ti:"${PORT}" 2>/dev/null); then
    echo "[$(date -Iseconds)] killing stale process on port ${PORT}: ${STALE}"
    kill -KILL ${STALE} 2>/dev/null || true
    sleep 2
fi

# Probe code-server responsiveness every 1s during the load, in background.
# Writes one CSV row per sample: ts, http_code, latency_ms. If latency
# spikes during load, the operator's UI was frozen.
CS_OUT="${OUTDIR}/code_server_probe.csv"
echo "ts,http_code,latency_ms" > "${CS_OUT}"
(
    while true; do
        T0=$(date +%s.%N)
        # Use code-server's own port (8080); the / endpoint redirects but
        # responds quickly when the process is responsive
        HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8080/ 2>/dev/null || echo "000")
        T1=$(date +%s.%N)
        LATENCY_MS=$(awk -v t0="${T0}" -v t1="${T1}" 'BEGIN{printf "%.0f", (t1-t0)*1000}')
        printf "%s,%s,%s\n" "${T0}" "${HTTP}" "${LATENCY_MS}" >> "${CS_OUT}"
        sleep 1
    done
) &
CS_PROBE_PID=$!
echo "[$(date -Iseconds)] code-server probe PID=${CS_PROBE_PID}"

# Hardcoded command for the preserve preset (mirrors the rendered unit but
# skips the ExecStartPre preflight, which is the whole point of this script).
LLAMA_BIN="$HOME/bin/llama.cpp/build/bin/llama-server"
LLAMA_ARGS=(
    --model "$HOME/bin/Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf"
    --port "${PORT}"
    --host 127.0.0.1
    --ctx-size 131072
    --n-gpu-layers -1
    --jinja
    --flash-attn on
    --parallel 1 --threads 16
    --cache-type-k q8_0 --cache-type-v q8_0
    --spec-type draft-mtp --spec-draft-n-max 2
    --chat-template-kwargs '{"preserve_thinking":true}'
)
printf 'command: %s' "${LLAMA_BIN}" > "${OUTDIR}/command.txt"
printf ' %q' "${LLAMA_ARGS[@]}" >> "${OUTDIR}/command.txt"
echo >> "${OUTDIR}/command.txt"

# Pre-load snapshot
{
    echo "## Pre-load"
    echo "ts=$(date -Iseconds)"
    grep -E "MemTotal|MemFree|MemAvailable|Cached|SwapFree|SwapTotal|Pinned|Mlocked" /proc/meminfo
    cat /proc/pressure/memory
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
} > "${OUTDIR}/pre.txt"

# Start llama-server in background. Use `exec` in a subshell so $! is the
# llama-server PID itself, not the shell wrapper.
START=$(date +%s.%N)
echo "[$(date -Iseconds)] starting llama-server directly..."
("${LLAMA_BIN}" "${LLAMA_ARGS[@]}" > "${OUTDIR}/llama_server.log" 2>&1) &
PID=$!
echo "[$(date -Iseconds)] PID=${PID}"
echo "${PID}" > "${OUTDIR}/pid.txt"
# Verify PID is actually llama-server (not a shell)
sleep 0.5
COMM=$(cat /proc/${PID}/comm 2>/dev/null || echo "GONE")
echo "[$(date -Iseconds)] PID=${PID} comm=${COMM}"
if [ "${COMM}" != "llama-server" ] && [ "${COMM}" != "GONE" ]; then
    # Find the actual llama-server child
    CHILD=$(pgrep -P "${PID}" -x llama-server | head -1)
    if [ -n "${CHILD}" ]; then
        PID="${CHILD}"
        echo "[$(date -Iseconds)] using llama-server child PID=${PID}"
        echo "${PID}" > "${OUTDIR}/pid.txt"
    fi
fi

# Attach sampler
python3 docs/research/20260525_oom_protection_findings/sampler.py \
    --pid "${PID}" \
    --out "${OUTDIR}/sampler.csv" \
    --duration 240 \
    --interval 0.1 &
SAMPLER_PID=$!
echo "[$(date -Iseconds)] sampler PID=${SAMPLER_PID}"

# Poll /health (max 120s)
HEALTH_AT=""
OUTCOME="UNKNOWN"
for i in $(seq 1 240); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        OUTCOME="EXIT_BEFORE_HEALTH"
        echo "[$(date -Iseconds)] llama-server PID ${PID} exited before /health" >&2
        break
    fi
    if curl -sf -o /dev/null --max-time 2 "http://localhost:${PORT}/health" 2>/dev/null; then
        HEALTH_AT=$(date +%s.%N)
        OUTCOME="HEALTHY"
        echo "[$(date -Iseconds)] /health=200 after ${i} polls"
        break
    fi
    sleep 0.5
done

# Hold for 5s post-health if alive, then stop
if [ "${OUTCOME}" = "HEALTHY" ]; then
    sleep 5
fi
kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true

# Stop code-server probe
kill "${CS_PROBE_PID}" 2>/dev/null || true
wait "${CS_PROBE_PID}" 2>/dev/null || true

if kill -0 "${PID}" 2>/dev/null; then
    echo "[$(date -Iseconds)] stopping llama-server PID ${PID}"
    kill -TERM "${PID}"
    for i in $(seq 1 30); do
        kill -0 "${PID}" 2>/dev/null || break
        sleep 0.5
    done
    kill -KILL "${PID}" 2>/dev/null || true
fi

# Timings
if [ -n "${HEALTH_AT}" ]; then
    LOAD_SEC=$(python3 -c "print(f'{${HEALTH_AT} - ${START}:.2f}')")
else
    LOAD_SEC="FAILED"
fi

{
    echo "## Post-load"
    echo "ts=$(date -Iseconds)"
    echo "outcome=${OUTCOME}"
    echo "load_seconds=${LOAD_SEC}"
    grep -E "MemTotal|MemFree|MemAvailable|Cached|SwapFree|SwapTotal|Pinned|Mlocked" /proc/meminfo
    cat /proc/pressure/memory
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv
    echo "last 20 lines of llama_server.log:"
    tail -20 "${OUTDIR}/llama_server.log"
} > "${OUTDIR}/post.txt"

echo "[$(date -Iseconds)] === DONE label=${LABEL} outcome=${OUTCOME} load=${LOAD_SEC}s ==="
echo "  sampler csv: ${OUTDIR}/sampler.csv ($(wc -l < ${OUTDIR}/sampler.csv 2>/dev/null || echo 0) rows)"

# Code-server probe summary
CS_ROWS=$(wc -l < "${CS_OUT}" 2>/dev/null || echo 0)
if [ "${CS_ROWS}" -gt 1 ]; then
    awk -F, 'NR>1 {
        n++;
        if ($2 != "200" && $2 != "302" && $2 != "301" && $2 != "303") errs++;
        if ($3+0 > max) max = $3+0;
        sum += $3+0;
    }
    END {
        printf "  code-server probe: %d samples, %d non-success, peak latency %.0fms, avg %.0fms\n", n, errs+0, max, (n>0?sum/n:0)
    }' "${CS_OUT}"
fi
