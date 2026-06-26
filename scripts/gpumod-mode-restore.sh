#!/usr/bin/env bash
# gpumod-mode-restore.sh — re-apply the persisted gpumod mode at boot.
#
# Tracked in two places (intentionally identical):
#   1. gpumod repo:  <gpumod-repo>/scripts/gpumod-mode-restore.sh  (canonical source)
#   2. k3s-setup:    gpu-services/gpumod/scripts/gpumod-mode-restore.sh
# This copy is what the gpumod-mode-restore.service unit points at via
# ExecStart, but the gpumod-repo copy is the canonical source. k3s-setup's
# install-units.sh syncs this file onto the gpumod-repo scripts/ path so
# the unit and the script it execs can never drift out of shape.
#
# Why this exists: gpumod's lifecycle is *ephemeral*. `gpumod mode switch`
# calls `systemctl --user start` (not `enable --now`), so a mode's service
# units never survive reboot on their own. The gpumod-mcp server only syncs
# YAML presets/modes into the SQLite DB at startup — it does NOT re-apply
# `current_mode`. The DB still records the last mode, but nothing boots it.
# Result after every reboot: gemma4 down, and any *externally* enabled unit
# (e.g. an orphaned vllm-embedding-code [Install] symlink) is the only thing
# running. This script closes that gap.
#
# It is self-describing: reads whatever `current_mode` the DB holds, so it
# stays correct after `gpumod mode switch <anything>`. No-op when no mode is
# persisted (e.g. a deliberately-`blank` boot) — never force-starts services.
#
# Ordering: must run AFTER gpumod-mcp.service has finished its YAML→DB sync,
# otherwise `mode status` can read a stale/empty DB. The unit waits on the
# MCP HTTP endpoint (127.0.0.1:8808) as the readiness gate, not just `After=`.
#
# Idempotent: `mode switch` is a no-op if the target services are already
# running, and stops orphan services from prior modes. Safe to re-run.

set -euo pipefail

# Machine-specific config lives in a gitignored env file next to this script
# (copy gpumod-mode-restore.env.example -> gpumod-mode-restore.env). Keeps host
# paths out of the tracked script; falls back to sensible $HOME defaults so the
# committed copy works out-of-box on a standard checkout.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
[[ -f "${_SCRIPT_DIR}/gpumod-mode-restore.env" ]] && source "${_SCRIPT_DIR}/gpumod-mode-restore.env"

GPUMOD_DIR="${GPUMOD_DIR:-$HOME/AI/gpumod}"
GPUMOD_BIN="${GPUMOD_BIN:-${GPUMOD_DIR}/.venv/bin/gpumod}"
MCP_HOST="127.0.0.1"
MCP_PORT="8808"
# Max time to wait for gpumod-mcp to finish its YAML→DB sync and come up.
READY_TIMEOUT="${GPUMOD_MCP_READY_TIMEOUT:-120}"

log() { printf '[gpumod-mode-restore] %s\n' "$*" >&2; }

die() { log "ERROR: $*"; exit 1; }

# --- 1. Wait for gpumod-mcp to be up (owns YAML→DB sync; ordering gate) ---
wait_for_mcp() {
    local deadline=$(( SECONDS + READY_TIMEOUT ))
    log "waiting for gpumod-mcp HTTP at ${MCP_HOST}:${MCP_PORT} (<= ${READY_TIMEOUT}s)"
    while (( SECONDS < deadline )); do
        # Bare /mcp returns 406 once the streamable-http endpoint is bound —
        # that's "up" for our purposes (we only need the DB synced, not a
        # handshake). Any TCP accept + HTTP response counts as ready.
        if curl -fsS -m 2 -o /dev/null "http://${MCP_HOST}:${MCP_PORT}/"; then
            log "gpumod-mcp is up (HTTP 2xx)"
            return 0
        fi
        # 406 is expected for the MCP endpoint — also means it's listening.
        local code
        code="$(curl -sS -m 2 -o /dev/null -w '%{http_code}' "http://${MCP_HOST}:${MCP_PORT}/mcp" 2>/dev/null || true)"
        if [[ "$code" =~ ^(200|406|400)$ ]]; then
            log "gpumod-mcp is up (HTTP ${code} on /mcp)"
            return 0
        fi
        sleep 1
    done
    die "gpumod-mcp did not become ready within ${READY_TIMEOUT}s — aborting mode restore"
}

# --- 2. Read persisted current_mode from the DB (self-describing) ---
read_current_mode() {
    # `mode status --json` prints the mode doc (with "id") when a current
    # mode is set, or {"active":false,"mode":null} when none. stdout only —
    # rich/status banners go to stderr.
    local status_json
    if ! status_json="$("$GPUMOD_BIN" mode status --json 2>/dev/null)"; then
        die "failed to query gpumod mode status — is gpumod-mcp healthy?"
    fi
    if [[ -z "$status_json" ]]; then
        die "gpumod mode status --json returned empty output"
    fi
    # Extract id; treat null/missing as "no persisted mode".
    "$GPUMOD_DIR"/.venv/bin/python3 -c '
import json, sys
try:
    data = json.loads(sys.stdin.read())
except Exception as exc:
    print(f"PARSE_ERROR: {exc}", file=sys.stderr); sys.exit(2)
mid = data.get("id") if isinstance(data, dict) else None
if not mid:
    print("NONE", end="")  # no persisted mode (e.g. blank boot)
else:
    print(mid, end="")
' <<<"$status_json"
}

# --- 3. Apply the mode (idempotent; starts targets, stops orphans) ---
apply_mode() {
    local mode_id="$1"
    log "re-applying persisted mode: ${mode_id}"
    if "$GPUMOD_BIN" mode switch "$mode_id" --json >/dev/null 2>&1; then
        log "mode '${mode_id}' re-applied successfully"
    else
        # Don't hard-fail the unit on a transient switch error; the next
        # reboot (or manual `gpumod mode switch <id>`) retries. Surface it.
        log "WARNING: mode switch to '${mode_id}' reported failure — check \`gpumod mode status\` and journalctl --user -u gpumod-mode-restore"
        return 1
    fi
}

main() {
    [[ -x "$GPUMOD_BIN" ]] || die "gpumod binary not found at ${GPUMOD_BIN}"
    command -v curl >/dev/null || die "curl not found (required for MCP readiness gate)"

    wait_for_mcp

    local mode_id
    mode_id="$(read_current_mode)" || die "could not read current mode from DB"

    if [[ "$mode_id" == "NONE" ]]; then
        log "no persisted current_mode in DB — nothing to restore (likely a blank boot). Exiting cleanly."
        exit 0
    fi

    apply_mode "$mode_id"
}

main "$@"
