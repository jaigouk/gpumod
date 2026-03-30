"""Spike Q1-Q4: Verify Apple Silicon GPU memory metrics.

Run on macOS Apple Silicon to answer:
  Q1. Are ioreg IOAccelerator PerformanceStatistics keys stable?
  Q2. What is the unified memory budget (recommendedMaxWorkingSetSize)?
  Q3. Does pyobjc-framework-Metal import without Xcode CLI tools?
  Q4. Which ioreg field correlates with memory pressure under load?

Usage:
    uv run python docs/internal/research/20260330_macos_spike/verify_gpu_metrics.py

Requires: macOS, Apple Silicon. Optional: pyobjc-framework-Metal.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def check_platform() -> bool:
    if sys.platform != "darwin":
        print("SKIP: Not macOS")
        return False
    if platform.machine() != "arm64":
        print("SKIP: Not Apple Silicon (arm64)")
        return False
    return True


def q1_ioreg_keys() -> dict:
    """Q1: Query ioreg IOAccelerator and report all PerformanceStatistics keys."""
    print("\n=== Q1: ioreg IOAccelerator PerformanceStatistics keys ===")
    import plistlib

    result = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", "-a"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: ioreg failed: {result.stderr.decode()}")
        return {}

    data = plistlib.loads(result.stdout)
    findings: dict = {}
    for entry in data:
        model = entry.get("model", "unknown")
        cores = entry.get("gpu-core-count", "unknown")
        ps = entry.get("PerformanceStatistics", {})
        if not ps:
            continue

        findings["model"] = model
        findings["gpu_core_count"] = cores
        findings["perf_stats_keys"] = sorted(ps.keys())

        # Key metrics for gpumod
        alloc = ps.get("Alloc system memory", "MISSING")
        in_use = ps.get("In use system memory", "MISSING")
        util = ps.get("Device Utilization %", "MISSING")

        findings["alloc_system_memory_bytes"] = alloc
        findings["in_use_system_memory_bytes"] = in_use
        findings["device_utilization_pct"] = util

        print(f"  Model: {model}")
        print(f"  GPU Cores: {cores}")
        print(f"  Alloc system memory: {alloc} bytes ({alloc // 1024 // 1024 if isinstance(alloc, int) else '?'} MB)")
        print(f"  In use system memory: {in_use} bytes ({in_use // 1024 // 1024 if isinstance(in_use, int) else '?'} MB)")
        print(f"  Device Utilization: {util}%")
        print(f"  All keys ({len(ps)}): {sorted(ps.keys())}")

        # Check for the keys gpumod needs
        required = ["Alloc system memory", "In use system memory", "Device Utilization %"]
        missing = [k for k in required if k not in ps]
        if missing:
            print(f"  WARNING: Missing required keys: {missing}")
        else:
            print("  OK: All required keys present")

    return findings


def q2_memory_budget() -> dict:
    """Q2: Determine unified memory budget via multiple methods."""
    print("\n=== Q2: Unified memory budget calculation ===")
    findings: dict = {}

    # Method 1: sysctl hw.memsize
    result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    total_bytes = int(result.stdout.strip())
    total_gb = total_bytes / (1024**3)
    findings["hw_memsize_bytes"] = total_bytes
    findings["hw_memsize_gb"] = round(total_gb, 1)
    print(f"  sysctl hw.memsize: {total_bytes} bytes ({total_gb:.1f} GB)")

    # Method 2: sysctl iogpu.wired_limit_mb
    result = subprocess.run(
        ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True
    )
    if result.returncode == 0:
        wired_limit = result.stdout.strip()
        findings["iogpu_wired_limit_mb"] = wired_limit
        print(f"  sysctl iogpu.wired_limit_mb: {wired_limit} (0 = system default)")
    else:
        findings["iogpu_wired_limit_mb"] = "NOT AVAILABLE"
        print(f"  sysctl iogpu.wired_limit_mb: NOT AVAILABLE ({result.stderr.strip()})")

    # Method 3: Compute expected threshold
    if total_gb >= 36:
        expected_pct = 0.75
    else:
        expected_pct = 0.66
    expected_gpu_mb = int(total_bytes * expected_pct / (1024**2))
    findings["expected_gpu_budget_mb"] = expected_gpu_mb
    findings["expected_pct"] = expected_pct
    print(f"  Expected GPU budget ({expected_pct*100:.0f}% of {total_gb:.0f} GB): {expected_gpu_mb} MB")

    # Method 4: pyobjc Metal (if available)
    try:
        import Metal  # type: ignore[import-untyped]

        device = Metal.MTLCreateSystemDefaultDevice()
        rec_max = device.recommendedMaxWorkingSetSize()
        cur_alloc = device.currentAllocatedSize()
        has_unified = device.hasUnifiedMemory()
        max_buf = device.maxBufferLength()

        findings["metal_recommended_max_bytes"] = rec_max
        findings["metal_recommended_max_mb"] = rec_max // (1024**2)
        findings["metal_current_alloc_bytes"] = cur_alloc
        findings["metal_has_unified_memory"] = has_unified
        findings["metal_max_buffer_bytes"] = max_buf

        actual_pct = rec_max / total_bytes
        findings["metal_actual_pct"] = round(actual_pct, 4)

        print(f"  Metal recommendedMaxWorkingSetSize: {rec_max} bytes ({rec_max // (1024**2)} MB)")
        print(f"  Metal actual % of RAM: {actual_pct*100:.1f}%")
        print(f"  Metal currentAllocatedSize: {cur_alloc} bytes")
        print(f"  Metal hasUnifiedMemory: {has_unified}")
        print(f"  Metal maxBufferLength: {max_buf} bytes ({max_buf // (1024**2)} MB)")

        # Compare expected vs actual
        diff_mb = abs(expected_gpu_mb - rec_max // (1024**2))
        print(f"  Delta (expected vs Metal): {diff_mb} MB")
    except ImportError:
        findings["metal_available"] = False
        print("  pyobjc-framework-Metal: NOT INSTALLED (install with: uv pip install pyobjc-framework-Metal)")

    return findings


def q3_pyobjc_install() -> dict:
    """Q3: Check pyobjc-framework-Metal availability and build requirements."""
    print("\n=== Q3: pyobjc-framework-Metal build requirements ===")
    findings: dict = {}

    # Check if already installed
    try:
        import Metal  # type: ignore[import-untyped]

        findings["installed"] = True
        import importlib.metadata

        version = importlib.metadata.version("pyobjc-framework-Metal")
        findings["version"] = version
        print(f"  Installed: yes (v{version})")
    except ImportError:
        findings["installed"] = False
        print("  Installed: no")

    # Check Xcode CLI tools
    result = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
    xcode_path = result.stdout.strip() if result.returncode == 0 else "NOT FOUND"
    findings["xcode_cli_path"] = xcode_path
    print(f"  Xcode CLI tools: {xcode_path}")

    # Check pip index for wheel availability
    result = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "pyobjc-framework-Metal"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Sanitize home paths from output
        output = result.stdout.strip()[:500].replace(str(Path.home()), "~")
        findings["pip_index_output"] = output
        print(f"  pip index: {output[:200]}")
    else:
        err = result.stderr.strip()[:200].replace(str(Path.home()), "~")
        findings["pip_index_output"] = f"ERROR: {err}"
        print(f"  pip index: error ({err[:100]})")

    return findings


def main() -> None:
    if not check_platform():
        sys.exit(0)

    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {sys.version}")

    all_findings: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
    }

    all_findings["q1_ioreg"] = q1_ioreg_keys()
    all_findings["q2_memory_budget"] = q2_memory_budget()
    all_findings["q3_pyobjc"] = q3_pyobjc_install()

    # Save results
    out_path = "docs/internal/research/20260330_macos_spike/gpu_metrics_results.json"
    with open(out_path, "w") as f:
        json.dump(all_findings, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
