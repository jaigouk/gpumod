#!/usr/bin/env bash
# gpumod-x7rv: install the host-protection drop-ins.
#
# Run with sudo. Each layer is independently installed and verified before
# proceeding to the next. If any step fails, abort and document the failure.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root (sudo $0)" >&2
    exit 1
fi

CFG_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "${CFG_DIR}/code-server@.service.d-10-oom-protect.conf" ]; then
    echo "ERROR: drop-in source not found at ${CFG_DIR}" >&2
    exit 1
fi

echo "=== Step 1: code-server@.service drop-in ==="
mkdir -p /etc/systemd/system/code-server@.service.d
install -m 0644 \
    "${CFG_DIR}/code-server@.service.d-10-oom-protect.conf" \
    /etc/systemd/system/code-server@.service.d/10-oom-protect.conf
echo "  installed: /etc/systemd/system/code-server@.service.d/10-oom-protect.conf"

echo "=== Step 2: systemd-oomd drop-in ==="
mkdir -p /etc/systemd/oomd.conf.d
install -m 0644 \
    "${CFG_DIR}/oomd.conf.d-gpumod.conf" \
    /etc/systemd/oomd.conf.d/gpumod.conf
echo "  installed: /etc/systemd/oomd.conf.d/gpumod.conf"

echo "=== Step 3: daemon-reload + restart affected services ==="
systemctl daemon-reload
systemctl restart code-server@<user>
systemctl restart systemd-oomd
echo "  restarted: code-server@<user>, systemd-oomd"

echo "=== Verify ==="
echo "code-server memory directives:"
systemctl show code-server@<user> \
    -p MemoryMin,MemoryLow,OOMScoreAdjust,ManagedOOMMemoryPressure,ManagedOOMSwap,Restart,RestartUSec
echo "code-server cgroup memory.min:"
cat /sys/fs/cgroup/system.slice/system-code\\x2dserver.slice/code-server@<user>.service/memory.min || true
echo "code-server cgroup memory.low:"
cat /sys/fs/cgroup/system.slice/system-code\\x2dserver.slice/code-server@<user>.service/memory.low || true
echo "systemd-oomd config:"
systemd-analyze cat-config systemd/oomd.conf | tail -20

echo "=== DONE ==="
echo "Next: reconnect to code-server, verify SSH still works,"
echo "then proceed to gpumod app.slice memory.max (Step 4 — separate script)."
