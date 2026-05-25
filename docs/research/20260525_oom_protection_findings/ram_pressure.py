#!/usr/bin/env python3
"""Allocate anonymous RAM to reduce MemAvailable to a target level.

Used for spike gpumod-x7rv Phase 4. Holds the memory until killed via SIGTERM
or SIGINT.

Usage:
  python3 ram_pressure.py --target-mb 15000

The script:
  - reads current MemAvailable from /proc/meminfo
  - computes how much to allocate to reach the target
  - allocates a single anonymous bytearray and touches every 4 KiB page so
    the kernel commits real pages (not lazy allocation)
  - sleeps until SIGTERM / Ctrl-C, then frees and exits cleanly

DO NOT run unattended. If you set target-mb too low (e.g. 500), the kernel
OOM-killer may fire and the system may freeze. Have drop_caches ready.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


def mem_available_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=int, required=True,
                    help="reduce MemAvailable to approximately this level")
    ap.add_argument("--safety-floor-mb", type=int, default=2000,
                    help="never allocate so much that MemAvailable drops below this (default 2000)")
    args = ap.parse_args()

    current = mem_available_mb()
    print(f"current MemAvailable: {current} MB")
    print(f"target MemAvailable:  {args.target_mb} MB", flush=True)

    needed = current - args.target_mb
    if needed <= 0:
        print(f"already at or below target (current {current} <= target {args.target_mb}); nothing to do",
              flush=True)
        return 0

    if args.target_mb < args.safety_floor_mb:
        print(f"REFUSED: target {args.target_mb} < safety floor {args.safety_floor_mb}",
              file=sys.stderr)
        return 1

    print(f"allocating {needed} MB...", flush=True)
    buf = bytearray(needed * 1024 * 1024)
    # Touch every page (4 KiB) to force kernel commit
    page = 4096
    for i in range(0, len(buf), page):
        buf[i] = 1

    final = mem_available_mb()
    print(f"allocated; MemAvailable now {final} MB ({needed} MB consumed)",
          flush=True)
    print(f"holding memory; send SIGTERM or Ctrl-C to release", flush=True)

    stop = False

    def handle(signum: int, _frame) -> None:
        nonlocal stop
        print(f"received signal {signum}; releasing", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    while not stop:
        time.sleep(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
