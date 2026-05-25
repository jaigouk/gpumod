#!/bin/bash
# gpumod-1lpe: install OOM protection drop-ins for code-server and systemd-oomd.
#
# Copies the drop-in configs to their systemd locations and restarts
# affected services. Idempotent — re-running overwrites with the same
# content and re-applies. Requires sudo.
#
# Usage:
#   sudo scripts/oom-protection/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CS_SRC="${SCRIPT_DIR}/code-server-protect.conf"
CS_DEST="/etc/systemd/system/code-server@.service.d/10-oom-protect.conf"

OOMD_SRC="${SCRIPT_DIR}/oomd-gpumod.conf"
OOMD_DEST="/etc/systemd/oomd.conf.d/gpumod.conf"

# --- Preflight checks ---

if [[ $EUID -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

if [[ ! -f "$CS_SRC" ]]; then
    echo "error: source file not found: $CS_SRC" >&2
    exit 1
fi

if [[ ! -f "$OOMD_SRC" ]]; then
    echo "error: source file not found: $OOMD_SRC" >&2
    exit 1
fi

# --- Install code-server drop-in ---

echo "=== Step 1: code-server drop-in ==="
mkdir -p "$(dirname "$CS_DEST")"
install -m 0644 "$CS_SRC" "$CS_DEST"
echo "  installed: $CS_DEST"

# --- Install oomd drop-in ---

echo "=== Step 2: systemd-oomd drop-in ==="
mkdir -p "$(dirname "$OOMD_DEST")"
install -m 0644 "$OOMD_SRC" "$OOMD_DEST"
echo "  installed: $OOMD_DEST"

# --- Reload and restart ---

echo "=== Step 3: daemon-reload + restart ==="
systemctl daemon-reload

# Restart all code-server instances (template unit)
# Find active instances and restart them
for unit in $(systemctl list-units --type=service --state=running --no-legend 'code-server@*' | awk '{print $1}'); do
    echo "  restarting: $unit"
    systemctl restart "$unit"
done

systemctl restart systemd-oomd
echo "  restarted: systemd-oomd"

# --- Verify ---

echo "=== Verify ==="
echo "code-server drop-in installed:"
cat "$CS_DEST" | grep -E '^(MemoryMin|MemoryLow|OOMScoreAdjust|ManagedOOMMemoryPressure|ManagedOOMSwap)='
echo ""
echo "oomd drop-in installed:"
cat "$OOMD_DEST" | grep -E '^Default'
echo ""

echo "Done. Verify with: gpumod doctor oom-protection"
