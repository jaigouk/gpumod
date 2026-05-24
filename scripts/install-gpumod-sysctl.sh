#!/bin/bash
# gpumod-ej0: install gpumod stability sysctl tunings.
#
# Copies scripts/gpumod-stability.conf to /etc/sysctl.d/ and applies it.
# Idempotent — re-running just overwrites with the same content and
# re-applies. Requires sudo.

set -euo pipefail

SRC="$(dirname "$0")/gpumod-stability.conf"
DEST="/etc/sysctl.d/99-gpumod-stability.conf"

if [[ ! -f "$SRC" ]]; then
    echo "error: source file not found: $SRC" >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "error: must run as root (use sudo)" >&2
    exit 1
fi

echo "Installing $DEST"
cp "$SRC" "$DEST"
chmod 644 "$DEST"

echo "Applying sysctl tunings"
sysctl --system >/dev/null

current=$(cat /proc/sys/vm/min_free_kbytes)
echo "vm.min_free_kbytes is now: $current kB"

if [[ "$current" -lt 1048576 ]]; then
    echo "warning: vm.min_free_kbytes did not take effect — expected >= 1048576" >&2
    exit 1
fi

echo "Done. Verify with: gpumod doctor sysctl"
