# Q2, Q3, Q4 Findings: GPU Memory Metrics on Apple Silicon

**Date:** 2026-03-30
**Machine:** Apple M2 Max, 32 GB unified memory, macOS 26.4, 30 GPU cores
**Python:** 3.12.7
**Branch:** macos

---

## Q2: Unified Memory Budget (pyobjc Metal)

### Key Finding: 78%, NOT 66%

The initial research hypothesized a 66% threshold for machines with < 36 GB RAM.
**The actual `recommendedMaxWorkingSetSize` is 78% of total RAM**, significantly
higher than the 66% estimate.

| Metric | Value |
|--------|-------|
| `hw.memsize` (total RAM) | 34,359,738,368 bytes (32 GB) |
| `iogpu.wired_limit_mb` | 0 (system default) |
| `recommendedMaxWorkingSetSize` | 26,800,603,136 bytes (25,559 MB) |
| Actual % of total RAM | **78.00%** |
| Expected 66% budget | 21,626 MB |
| Expected 75% budget | 24,576 MB |
| Delta from 66% estimate | **+3,933 MB** (Metal allows ~4 GB more than expected) |
| Delta from 75% estimate | **+983 MB** |
| `maxBufferLength` | 20,100,448,256 bytes (19,169 MB = ~58.5% of RAM) |
| `currentAllocatedSize` | 65,536 bytes (64 KB, process-only) |
| `hasUnifiedMemory` | True |

### Analysis

The 66%/75% thresholds cited in community sources (e.g., stencel.io, Eclectic Light)
are **approximations from 2022-era macOS** (Monterey, Ventura). Apple has
apparently raised the limit in macOS 26.x. The 78% value is consistent
with llama.cpp's own Metal init output, which reported
`recommendedMaxWorkingSetSize = 26800.60 MB` during our Q4 experiment.

**Implication for gpumod:** Do NOT hardcode 66% or 75%. Always query
`recommendedMaxWorkingSetSize` via the Metal API (pyobjc) or derive from
ioreg. The threshold may change across macOS versions and chip generations.

### iogpu.wired_limit_mb Override

The sysctl variable `iogpu.wired_limit_mb` can override the GPU memory limit:

```
sudo sysctl -w iogpu.wired_limit_mb=8192   # set to 8 GB
sudo sysctl -w iogpu.wired_limit_mb=0      # restore default
```

- Requires sudo privileges
- Takes effect immediately (no reboot needed)
- Current value: 0 (default)
- **Not tested** whether this changes `recommendedMaxWorkingSetSize` (requires sudo)
- Documented for reference; gpumod should check this value and warn if non-zero

**Source:** Verified on this machine 2026-03-30. Apple Developer docs:
[recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize),
[sysctl iogpu.wired_limit_mb](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe)

---

## Q3: pyobjc-framework-Metal Wheel Check

### Answer: Pre-built wheels. No Xcode required for installation.

| Check | Result |
|-------|--------|
| Installation method | Pre-built binary wheel (`.whl`) |
| Install time | 5ms (3 packages: pyobjc-core, pyobjc-framework-Cocoa, pyobjc-framework-Metal) |
| Source build needed? | **No** |
| Xcode CLI tools required? | **No** (for installation; present on this machine but not needed) |
| Wheel platform tag | `cp312-cp312-macosx_10_13_universal2` |
| Package version | 12.1 |
| Total install size | ~5 MB (3 packages) |

### Wheel Availability on PyPI (v12.1)

Pre-built wheels available for Python 3.10 through 3.14:

| Python | Platform Tag |
|--------|-------------|
| 3.10 | `cp310-cp310-macosx_10_9_universal2` |
| 3.11 | `cp311-cp311-macosx_10_9_universal2` |
| 3.12 | `cp312-cp312-macosx_10_13_universal2` |
| 3.13 | `cp313-cp313-macosx_10_13_universal2` |
| 3.13t | `cp313-cp313t-macosx_10_13_universal2` |
| 3.14 | `cp314-cp314-macosx_10_15_universal2` |
| 3.14t | `cp314-cp314t-macosx_10_15_universal2` |

A source distribution (`.tar.gz`) is also available on PyPI but was NOT used
during installation since the wheel matched.

### Dependency Risk Assessment

- **Low risk.** Pre-built wheels cover Python 3.10-3.14 on macOS universal2.
- pyobjc is actively maintained by Ronald Oussoren (12.1 released Nov 2025).
- MIT license -- permissive.
- No C compilation needed at install time when wheel matches.
- If a future Python version lacks a wheel (unlikely, given 3.14t already covered),
  building from source would require Xcode CLI tools.

**Source:** [PyPI pyobjc-framework-Metal 12.1](https://pypi.org/project/pyobjc-framework-Metal/12.1/),
verified installation on this machine 2026-03-30.

---

## Q4: Memory Semantics Under ML Workload

### Methodology

Measured ioreg `IOAccelerator PerformanceStatistics` at four phases using
llama-server with a tiny model (stories260K, 1.1 MB, 293K params) and
`--n-gpu-layers 99` (fully offloaded to Metal).

The model is too small to show dramatic memory changes, but the methodology
is validated. Results below demonstrate the measurement pipeline works.

### Memory Semantics Table (stories260K, 1.1 MB model)

| Phase | Alloc system memory | In use system memory | Delta (Alloc - In use) | Device Util % |
|-------|-------------------:|--------------------:|---------------------:|--------------:|
| **Baseline** (no llama-server) | 1,533 MB | 299 MB | 1,233 MB | 9% |
| **Model loaded** (idle) | 1,707 MB | 469 MB | 1,238 MB | 20% |
| **After generation** (32 tokens) | 1,698 MB | 277 MB | 1,421 MB | 64% |
| **After server exit** (3s wait) | 1,542 MB | 589 MB | 953 MB | 2% |

### Observations

1. **"Alloc system memory" increased ~174 MB on model load** (1533 -> 1707 MB),
   far more than the 1.1 MB model size. This includes Metal buffers,
   KV cache (1.25 MB), and compute buffers (36.5 MB) reported by llama.cpp.

2. **"In use system memory" is volatile.** It dropped from 469 MB (model loaded)
   to 277 MB (after generation), likely because Metal reclaimed compute buffers
   post-inference. This is NOT a reliable measure of allocated model memory.

3. **"Alloc system memory" is the correct metric for gpumod's VRAMTracker.**
   It tracks total GPU allocations system-wide and moves monotonically with
   model load/unload (up on load, down after exit).

4. **Memory release is fast for small models.** After server exit, Alloc dropped
   within 3 seconds. For large models (7B+), the CUDA-like async release race
   condition documented in the architecture (`VRAMTracker.wait_for_vram`) applies
   -- Metal memory may take seconds to fully release.

5. **Device Utilization spiked to 64% during generation**, even for a tiny model.
   This metric is useful for load detection but not for memory tracking.

### llama.cpp Metal Init Output (Confirms Metal API Values)

During model load, llama.cpp logged:
```
ggml_metal_init: GPU name:   Apple M2 Max
ggml_metal_init: GPU family: MTLGPUFamilyApple8 (1008)
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 26800.60 MB
```

This confirms llama.cpp sees the same 25,559 MB / 78% budget as our pyobjc query.

### What a Real Workload Test Requires

A 15 GB Q4_K_M GGUF model is available on this machine. To measure meaningful
memory deltas:

```bash
# Start server with large model (replace <path> with actual GGUF path)
/opt/homebrew/bin/llama-server --model <path-to-Q4_K_M.gguf> --port 18080 --n-gpu-layers 99

# Measure ioreg at each phase:
# 1. Baseline (before server start)
# 2. Model loaded (after /health returns ok)
# 3. During generation (concurrent with inference)
# 4. After server exit (poll ioreg every 1s for 30s to measure release time)
```

Expected: "Alloc system memory" should jump by ~15 GB on model load and
return to baseline within seconds of server exit.

### Recommendation for VRAMTracker on macOS

```
GPU capacity:        recommendedMaxWorkingSetSize (via pyobjc Metal)
GPU current usage:   ioreg "Alloc system memory" (via plistlib, stdlib)
GPU free:            capacity - current_usage
Wait-for-release:   poll ioreg "Alloc system memory" until it drops below threshold
```

The "In use system memory" field is too volatile for reliable tracking.
Use "Alloc system memory" as the macOS equivalent of `nvidia-smi` memory usage.

---

## Surprises and Escalation-Worthy Findings

### 1. CRITICAL: 78% threshold invalidates 66% assumption

The initial research document and community sources suggest 66% for < 36 GB
machines. The actual Metal API reports 78%. **Any code that hardcodes 66% or
75% will undercount available GPU memory by 2-4 GB.** Always query the API.

### 2. "In use system memory" is NOT what it sounds like

Despite the name, "In use system memory" fluctuates even when a model is
loaded and idle. It appears to track actively-in-flight GPU operations,
not total allocated memory. For gpumod, use "Alloc system memory" instead.

### 3. pyobjc is a trivially safe dependency

5 MB total, pre-built wheels for Python 3.10-3.14, MIT license, actively
maintained. No Xcode build step. The dependency risk is negligible compared
to the value of getting `recommendedMaxWorkingSetSize` without guessing.

---

## Summary for gpumod Implementation

| Question | Answer |
|----------|--------|
| Q2: What is the GPU memory budget? | 25,559 MB (78% of 32 GB RAM), via `recommendedMaxWorkingSetSize` |
| Q2: Is 66%/75% accurate? | **No.** Actual is 78% on macOS 26.4 / M2 Max. Do not hardcode. |
| Q2: Can `iogpu.wired_limit_mb` override it? | Yes (sudo required). Not tested but documented. |
| Q3: Wheel or source build? | Pre-built wheel (`cp312-cp312-macosx_10_13_universal2`) |
| Q3: Xcode CLI tools required? | No (for wheel install) |
| Q3: Dependency risk? | Low (MIT, 5 MB, wheels for 3.10-3.14, active maintainer) |
| Q4: Which ioreg field tracks GPU memory? | "Alloc system memory" (NOT "In use system memory") |
| Q4: How fast does memory release? | < 3s for tiny model; unknown for large models (needs testing) |
| Q4: Does llama.cpp agree with Metal API? | Yes -- same 26800.60 MB budget reported |
