#!/usr/bin/env python3
"""10 Hz memory sampler for gpumod-x7rv spike.

Tracks /proc/<pid>/status, /proc/meminfo, /proc/pressure/memory, and
nvidia-smi memory.used at a fixed cadence and writes a tidy CSV.

Usage:
  python3 sampler.py --pid <PID> --out runs/<NAME>.csv [--duration 300]

The script captures only the metrics relevant to the spike:
  ts, mem_avail_mb, mem_free_mb, cached_mb, swap_free_mb,
  pinned_kb, mlocked_kb, psi_some_avg10, psi_full_avg10,
  vmrss_mb, vmpin_mb, vmswap_mb, vmlck_mb,
  vram_used_mb, vram_free_mb

Stops when --duration elapses, the target PID disappears, or Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

NVIDIA_SMI = shutil.which("nvidia-smi")


def read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of int kB values."""
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip(":")
            try:
                out[key] = int(parts[1])
            except ValueError:
                continue
    return out


def read_psi() -> tuple[float, float]:
    """Return (some_avg10, full_avg10) from /proc/pressure/memory."""
    some = full = 0.0
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                if line.startswith("some "):
                    for tok in line.split():
                        if tok.startswith("avg10="):
                            some = float(tok.split("=", 1)[1])
                elif line.startswith("full "):
                    for tok in line.split():
                        if tok.startswith("avg10="):
                            full = float(tok.split("=", 1)[1])
    except OSError:
        pass
    return some, full


def read_proc_status(pid: int) -> dict[str, int]:
    """Parse /proc/<pid>/status, returning kB values for memory fields."""
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                if key in {"VmRSS", "VmPin", "VmSwap", "VmLck", "VmData"}:
                    try:
                        out[key] = int(parts[1])
                    except ValueError:
                        continue
    except FileNotFoundError:
        return {}
    return out


def read_nvidia_smi() -> tuple[int, int]:
    """Return (memory.used, memory.free) in MiB from nvidia-smi."""
    if NVIDIA_SMI is None:
        return 0, 0
    try:
        result = subprocess.run(
            [
                NVIDIA_SMI,
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        line = result.stdout.strip().splitlines()[0]
        used_s, free_s = line.split(",")
        return int(used_s), int(free_s)
    except (OSError, ValueError, IndexError):
        return 0, 0


def sample_once(pid: int) -> dict[str, int | float | str]:
    """Capture one sample as a flat dict."""
    meminfo = read_meminfo()
    psi_some, psi_full = read_psi()
    status = read_proc_status(pid)
    vram_used, vram_free = read_nvidia_smi()
    return {
        "ts": time.time(),
        "mem_avail_mb": meminfo.get("MemAvailable", 0) // 1024,
        "mem_free_mb": meminfo.get("MemFree", 0) // 1024,
        "cached_mb": meminfo.get("Cached", 0) // 1024,
        "buffers_mb": meminfo.get("Buffers", 0) // 1024,
        "swap_free_mb": meminfo.get("SwapFree", 0) // 1024,
        "swap_total_mb": meminfo.get("SwapTotal", 0) // 1024,
        "pinned_kb": meminfo.get("Pinned", 0),
        "mlocked_kb": meminfo.get("Mlocked", 0),
        "psi_some_avg10": psi_some,
        "psi_full_avg10": psi_full,
        "vmrss_mb": status.get("VmRSS", 0) // 1024,
        "vmpin_mb": status.get("VmPin", 0) // 1024,
        "vmswap_mb": status.get("VmSwap", 0) // 1024,
        "vmlck_mb": status.get("VmLck", 0) // 1024,
        "vram_used_mb": vram_used,
        "vram_free_mb": vram_free,
    }


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--interval", type=float, default=0.1)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fields = list(sample_once(args.pid).keys())
    if not Path(f"/proc/{args.pid}").exists():
        print(f"PID {args.pid} not running", file=sys.stderr)
        return 1

    start = time.time()
    next_tick = start
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        try:
            while time.time() - start < args.duration:
                if not pid_alive(args.pid):
                    print(f"PID {args.pid} exited at t={time.time() - start:.2f}s")
                    break
                writer.writerow(sample_once(args.pid))
                fh.flush()
                next_tick += args.interval
                sleep_for = next_tick - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.time()
        except KeyboardInterrupt:
            print("sampler interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
