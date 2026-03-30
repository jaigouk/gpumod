"""Spike Q11: Verify RAM check portability options.

Compare approaches for cross-platform RAM detection to replace /proc/meminfo:
  - psutil.virtual_memory() (external dependency)
  - sysctl hw.memsize + vm_stat (macOS native, no deps)
  - os.sysconf (POSIX, limited)

Usage:
    uv run python docs/internal/research/20260330_macos_spike/verify_ram_portability.py

Works on both Linux and macOS.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys


def method_procmeminfo() -> dict:
    """Linux /proc/meminfo (current RAMCheck approach)."""
    print("\n=== Method 1: /proc/meminfo (Linux only) ===")
    from pathlib import Path

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        print("  NOT AVAILABLE (not Linux)")
        return {"available": False, "platform": "linux_only"}

    values = {}
    for line in meminfo.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0].rstrip(":")] = int(parts[1])

    total_mb = values.get("MemTotal", 0) // 1024
    available_mb = values.get("MemAvailable", 0) // 1024
    swap_free_mb = values.get("SwapFree", 0) // 1024

    result = {
        "available": True,
        "total_mb": total_mb,
        "available_mb": available_mb,
        "swap_free_mb": swap_free_mb,
        "has_mem_available": "MemAvailable" in values,
    }
    print(f"  Total: {total_mb} MB, Available: {available_mb} MB, SwapFree: {swap_free_mb} MB")
    return result


def method_psutil() -> dict:
    """psutil.virtual_memory() (cross-platform)."""
    print("\n=== Method 2: psutil.virtual_memory() (cross-platform) ===")
    try:
        import psutil

        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        result = {
            "available": True,
            "psutil_version": psutil.__version__,
            "total_mb": vm.total // (1024 * 1024),
            "available_mb": vm.available // (1024 * 1024),
            "used_mb": vm.used // (1024 * 1024),
            "percent": vm.percent,
            "swap_total_mb": swap.total // (1024 * 1024),
            "swap_free_mb": swap.free // (1024 * 1024),
        }
        print(f"  psutil v{psutil.__version__}")
        print(f"  Total: {result['total_mb']} MB, Available: {result['available_mb']} MB")
        print(f"  Used: {result['used_mb']} MB ({result['percent']}%)")
        print(f"  Swap: {result['swap_total_mb']} MB total, {result['swap_free_mb']} MB free")
        return result
    except ImportError:
        print("  NOT INSTALLED (install with: uv pip install psutil)")
        return {"available": False, "note": "psutil not installed"}


def method_sysctl() -> dict:
    """sysctl + vm_stat (macOS native)."""
    print("\n=== Method 3: sysctl + vm_stat (macOS only) ===")
    if sys.platform != "darwin":
        print("  NOT AVAILABLE (not macOS)")
        return {"available": False, "platform": "macos_only"}

    result: dict = {"available": True}

    # Total RAM
    r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    total_bytes = int(r.stdout.strip())
    result["total_mb"] = total_bytes // (1024 * 1024)
    print(f"  sysctl hw.memsize: {result['total_mb']} MB")

    # vm_stat for available memory
    r = subprocess.run(["vm_stat"], capture_output=True, text=True)
    vm_stat = {}
    page_size = 4096  # default
    for line in r.stdout.splitlines():
        if "page size of" in line:
            page_size = int(line.split()[-2])
        elif ":" in line:
            key, val = line.rsplit(":", 1)
            val = val.strip().rstrip(".")
            try:
                vm_stat[key.strip()] = int(val)
            except ValueError:
                pass

    free_pages = vm_stat.get("Pages free", 0)
    inactive_pages = vm_stat.get("Pages inactive", 0)
    speculative_pages = vm_stat.get("Pages speculative", 0)
    # "Available" on macOS ≈ free + inactive + speculative (rough equivalent)
    available_pages = free_pages + inactive_pages + speculative_pages
    available_mb = (available_pages * page_size) // (1024 * 1024)

    result["page_size"] = page_size
    result["free_mb"] = (free_pages * page_size) // (1024 * 1024)
    result["inactive_mb"] = (inactive_pages * page_size) // (1024 * 1024)
    result["available_mb_estimate"] = available_mb
    result["note"] = "available = free + inactive + speculative (rough MemAvailable equivalent)"
    print(f"  Free: {result['free_mb']} MB, Inactive: {result['inactive_mb']} MB")
    print(f"  Available (estimated): {available_mb} MB")

    return result


def method_os_sysconf() -> dict:
    """os.sysconf (POSIX, total only)."""
    print("\n=== Method 4: os.sysconf (POSIX) ===")
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_mb = (page_size * phys_pages) // (1024 * 1024)

        result = {
            "available": True,
            "total_mb": total_mb,
            "page_size": page_size,
            "has_avail_pages": hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names,
        }

        if result["has_avail_pages"]:
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            result["available_mb"] = (page_size * avail_pages) // (1024 * 1024)
            print(f"  Total: {total_mb} MB, Available: {result['available_mb']} MB")
        else:
            result["available_mb"] = None
            print(f"  Total: {total_mb} MB, Available: NOT SUPPORTED (no SC_AVPHYS_PAGES)")

        return result
    except (ValueError, OSError) as e:
        print(f"  NOT AVAILABLE: {e}")
        return {"available": False, "error": str(e)}


def main() -> None:
    print(f"Platform: {platform.platform()}")
    print(f"System: {sys.platform}")

    all_findings: dict = {
        "platform": platform.platform(),
        "system": sys.platform,
    }

    all_findings["procmeminfo"] = method_procmeminfo()
    all_findings["psutil"] = method_psutil()
    all_findings["sysctl_vmstat"] = method_sysctl()
    all_findings["os_sysconf"] = method_os_sysconf()

    # Comparison
    print("\n=== Comparison ===")
    print(f"  {'Method':<25} {'Total MB':>10} {'Available MB':>14} {'Cross-platform':>15}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*15}")
    for name, data in [
        ("/proc/meminfo", all_findings["procmeminfo"]),
        ("psutil", all_findings["psutil"]),
        ("sysctl+vm_stat", all_findings["sysctl_vmstat"]),
        ("os.sysconf", all_findings["os_sysconf"]),
    ]:
        if not data.get("available"):
            print(f"  {name:<25} {'N/A':>10} {'N/A':>14} {'N/A':>15}")
            continue
        total = data.get("total_mb", "?")
        avail = data.get("available_mb") or data.get("available_mb_estimate") or "?"
        xplat = "Yes" if name == "psutil" else ("POSIX" if name == "os.sysconf" else "No")
        print(f"  {name:<25} {total:>10} {avail:>14} {xplat:>15}")

    # Recommendation
    print("\n=== Recommendation for Q11 ===")
    if sys.platform == "darwin":
        psutil_avail = all_findings["psutil"].get("available", False)
        if psutil_avail:
            psutil_mb = all_findings["psutil"]["available_mb"]
            vmstat_mb = all_findings["sysctl_vmstat"].get("available_mb_estimate", "?")
            print(f"  psutil available_mb: {psutil_mb}")
            print(f"  vm_stat estimated:   {vmstat_mb}")
            if isinstance(vmstat_mb, int):
                diff = abs(psutil_mb - vmstat_mb)
                print(f"  Difference: {diff} MB")

        print("\n  Options:")
        print("  1. psutil.virtual_memory() — simplest, cross-platform, but adds ~3MB dependency")
        print("  2. MemoryInfoProvider protocol — /proc/meminfo on Linux, sysctl+vm_stat on macOS")
        print("  3. os.sysconf — POSIX but SC_AVPHYS_PAGES may not exist on macOS")

    all_findings["recommendation"] = (
        "psutil.virtual_memory() is the simplest cross-platform solution. "
        "It provides total, available, used, and percent on both Linux and macOS. "
        "Adding psutil (~3MB) avoids maintaining platform-specific parsing code. "
        "If avoiding the dependency is important, use a MemoryInfoProvider protocol "
        "with /proc/meminfo on Linux and sysctl+vm_stat on macOS."
    )
    print(f"\n  {all_findings['recommendation']}")

    out_path = "docs/internal/research/20260330_macos_spike/ram_portability_results.json"
    with open(out_path, "w") as f:
        json.dump(all_findings, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
