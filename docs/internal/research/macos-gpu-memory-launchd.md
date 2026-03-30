# macOS Apple Silicon GPU Memory and launchd Research

**Date:** 2026-03-30
**Status:** Complete
**Branch:** macos

---

## 1. GPU Memory Metrics on Apple Silicon

### Summary of Options

| Option | Feasibility | Accuracy | Dependencies | Sudo | Maintenance |
|--------|-------------|----------|--------------|------|-------------|
| **pyobjc + Metal** | High | High (per-process Metal allocations) | pyobjc-framework-Metal | No | Active (v12.1, Nov 2025) |
| **MLX metal APIs** | High | MLX-only allocations | mlx (large, 50+ MB) | No | Active (v0.31.1, Apple-maintained) |
| **ioreg + plistlib** | High | High (system-wide GPU memory) | None (stdlib) | No | macOS built-in |
| **subprocess + Metal (pyobjc)** | N/A | N/A | N/A | N/A | N/A |
| **psutil** | Low | No GPU data | psutil | No | Active but no GPU support |
| **asitop** | Low | No memory data | powermetrics | Yes (sudo) | Dormant (last commit Oct 2021) |
| **macmon** | Medium | Power/utilization only, no memory | Rust binary | No | Active (MIT) |
| **apple-gpu** | Medium | Unknown | apple-gpu | No | Dormant (v0.3.0, Oct 2023, MIT) |
| **system_profiler** | Low | Static info only (model, cores) | None (stdlib) | No | macOS built-in |

### Recommendation: ioreg + plistlib (primary) + pyobjc Metal (optional)

---

### Option 1: ioreg + plistlib (RECOMMENDED - Primary)

**Verified on this machine (Apple M2 Max, 32 GB, macOS 15).**

```python
import subprocess, plistlib

result = subprocess.run(
    ['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator', '-a'],
    capture_output=True, text=False
)
data = plistlib.loads(result.stdout)
for entry in data:
    if 'PerformanceStatistics' in entry:
        ps = entry['PerformanceStatistics']
        model = entry.get('model', 'unknown')
        alloc_bytes = ps.get('Alloc system memory', 0)
        in_use_bytes = ps.get('In use system memory', 0)
        cores = entry.get('gpu-core-count', 0)
```

**Live test output:**

```
Model: Apple M2 Max
GPU Cores: 30
Alloc system memory: 2074771456 bytes = 1978 MB
In use system memory: 761839616 bytes = 726 MB
```

**Available PerformanceStatistics keys:**
- `Alloc system memory` -- total GPU memory allocated (all processes)
- `In use system memory` -- GPU memory actively in use
- `In use system memory (driver)` -- driver-level allocation
- `Device Utilization %` -- GPU utilization
- `Renderer Utilization %`, `Tiler Utilization %` -- subsystem utilization
- `recoveryCount`, `lastRecoveryTime` -- GPU crash recovery info

**For total GPU capacity**, combine with sysctl:
- `hw.memsize` = total physical RAM (34359738368 = 32 GB)
- `iogpu.wired_limit_mb` = GPU wired memory limit (0 = default)
- Default GPU limit: 75% of RAM for >= 36 GB systems, ~66% for < 36 GB

**Pros:**
- Zero external dependencies (subprocess + plistlib are stdlib)
- No sudo required
- Reports system-wide GPU memory (all processes, not just ours)
- Plist XML output is structured and easy to parse
- Includes GPU model name, core count, utilization

**Cons:**
- `ioreg` subprocess call (~50-100ms latency)
- IOAccelerator keys are undocumented (private API, may change across macOS versions)
- Does not report per-process GPU memory breakdown
- "Alloc system memory" vs "In use system memory" semantics are not formally documented

**Source:** Verified live on this machine. IOAccelerator PerformanceStatistics documented in community posts at [MacRumors Forums](https://forums.macrumors.com/threads/request-share-your-ioreg-ioaccelerator-results-please.2293664/) and [Eclectic Light](https://eclecticlight.co/2022/03/01/making-sense-of-m1-memory-use/).

---

### Option 2: pyobjc + Metal Framework

**Verified on this machine with pyobjc-framework-Metal 12.1.**

```python
import Metal

device = Metal.MTLCreateSystemDefaultDevice()
device.name()                           # "Apple M2 Max"
device.recommendedMaxWorkingSetSize()   # 26800603136 bytes (25559 MB)
device.currentAllocatedSize()           # bytes allocated by THIS process
device.hasUnifiedMemory()               # True
device.maxBufferLength()                # 20100448256 bytes (19169 MB)
```

**Live test output:**

```
Device name: Apple M2 Max
recommendedMaxWorkingSetSize: 26800603136 bytes = 25559 MB
currentAllocatedSize: 65536 bytes = 0 MB  (our process only)
hasUnifiedMemory: True
maxBufferLength: 20100448256 bytes = 19169 MB
```

**Key properties:**
- `recommendedMaxWorkingSetSize` -- Apple's recommended GPU memory limit (~75% of RAM)
- `currentAllocatedSize` -- Metal memory allocated by the calling process only
- `hasUnifiedMemory` -- True on all Apple Silicon
- `maxBufferLength` -- max single buffer size

**Package details:**
- **Version:** 12.1 (released November 14, 2025)
- **License:** MIT
- **Python:** >= 3.10
- **Dependencies:** pyobjc-core, pyobjc-framework-Cocoa
- **Size:** ~5 MB total install
- **Source:** [PyPI](https://pypi.org/project/pyobjc-framework-Metal/), [Docs](https://pyobjc.readthedocs.io/en/latest/apinotes/Metal.html)

**Pros:**
- Official Apple API via Python bindings
- `recommendedMaxWorkingSetSize` is the correct way to get GPU memory capacity
- No subprocess overhead
- Well-maintained by Ronald Oussoren

**Cons:**
- `currentAllocatedSize` only shows THIS process's Metal allocations (not system-wide)
- Cannot see other processes' GPU memory usage
- Adds ~5 MB dependency (pyobjc-core + Cocoa + Metal)
- macOS only (not a problem for this use case)

**Source:** [Apple Developer - currentAllocatedSize](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize), [Apple Developer - recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

---

### Option 3: MLX Metal APIs

**Verified on this machine with mlx 0.31.1.**

```python
import mlx.core as mx

info = mx.device_info()  # mx.metal.device_info() is deprecated
# Returns:
#   architecture: applegpu_g14s
#   device_name: Apple M2 Max
#   max_buffer_length: 20100448256
#   max_recommended_working_set_size: 26800603136
#   memory_size: 34359738368  <-- total physical RAM
#   resource_limit: 499000

mx.get_active_memory()   # MLX-managed memory only
mx.get_peak_memory()     # Peak MLX memory
mx.get_cache_memory()    # MLX cache memory
```

**Pros:**
- `memory_size` gives total physical RAM directly
- `max_recommended_working_set_size` matches Metal's API
- Active Apple-maintained project (WWDC 2025 featured)

**Cons:**
- Heavy dependency (~50+ MB including mlx-metal)
- Memory functions only track MLX's own allocations, not system-wide GPU usage
- Overkill just for memory queries
- `mx.metal.device_info()` already deprecated (use `mx.device_info()`)
- Requires Apple Silicon (no Intel Mac support)

**Source:** [MLX Unified Memory docs](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html), [MLX Metal docs](https://ml-explore.github.io/mlx/build/html/python/metal.html)

---

### Option 4: psutil on macOS

psutil reports system RAM and swap on macOS but has **no GPU-specific metrics**. `psutil.virtual_memory()` shows total/available/used system RAM. On Apple Silicon with unified memory, some of this RAM is used by the GPU, but psutil cannot distinguish GPU vs CPU usage.

**Verdict:** Useful for system RAM (replacing `/proc/meminfo` on Linux), but not for GPU memory.

**Source:** [psutil docs](https://psutil.readthedocs.io/)

---

### Option 5: llama.cpp Metal Memory

llama.cpp's server does not expose Metal memory usage via its HTTP API (`/health`, `/slots`, `/metrics`). The server reports:
- Health status
- Slot availability
- Token generation metrics

Metal memory is managed internally by ggml-metal. On macOS 15+, llama.cpp uses Metal residency sets to keep GPU memory wired. The 75% RAM limit is enforced by the Metal driver, not by llama.cpp.

**For gpumod's purposes:** Cannot query llama.cpp for its GPU memory usage. Must query the system directly via ioreg or Metal API.

**Source:** [llama.cpp Metal Backend](https://deepwiki.com/ggml-org/llama.cpp/5.2-http-server), [Apple Silicon limitations](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)

---

### Combined Strategy for gpumod macOS

```
GPU total capacity:
  pyobjc Metal: recommendedMaxWorkingSetSize (25559 MB on 32 GB)
  OR sysctl hw.memsize * 0.75 (for >= 36 GB) or * 0.66 (for < 36 GB)
  OR mlx device_info()['max_recommended_working_set_size']

GPU current usage (system-wide):
  ioreg IOAccelerator PerformanceStatistics:
    "Alloc system memory" = total allocated
    "In use system memory" = actively used

GPU free:
  recommendedMaxWorkingSetSize - "In use system memory"

System RAM:
  psutil.virtual_memory() (replaces /proc/meminfo)
```

---

## 2. launchd Plist Generation from Python

### Available Libraries

| Library | Purpose | Version | License | Maintained |
|---------|---------|---------|---------|------------|
| **plistlib** (stdlib) | Read/write Apple plist files | Python 3.12+ built-in | PSF | Yes (stdlib) |
| **launchd** (python-launchd) | Query/load/unload launchd jobs | 0.3.0 (Jun 2021) | MIT | Dormant — last release 2021 |
| **Jinja2** (already used) | Template plist XML | 3.x | BSD-3 | Yes |

**Recommendation:** Use `plistlib` (stdlib) for plist generation, or Jinja2 templates (already in the project for systemd unit files) for consistency. Use `subprocess` + `launchctl` for runtime interaction (load/unload/list) — more reliable than the dormant python-launchd package.

**Note:** `python-launchd` (PyPI: `launchd`) has not been updated since 2021 (v0.3.0). Do not depend on it for production use. Use subprocess calls to `launchctl` directly instead.

### plistlib Approach

```python
import plistlib
from pathlib import Path

plist = {
    'Label': 'com.gpumod.vllm-chat',
    'ProgramArguments': ['/opt/gpumod/venv/bin/python', '-m', 'vllm.entrypoints.openai.api_server', ...],
    'KeepAlive': True,
    'RunAtLoad': False,
    'EnvironmentVariables': {
        'CUDA_VISIBLE_DEVICES': '0',
        'MODEL_PATH': '/models/chat',
    },
    'StandardOutPath': '/tmp/gpumod-vllm-chat.stdout.log',
    'StandardErrorPath': '/tmp/gpumod-vllm-chat.stderr.log',
    'WorkingDirectory': '/opt/gpumod',
    'ThrottleInterval': 10,
    'ProcessType': 'Background',
}

plist_path = Path.home() / 'Library/LaunchAgents/com.gpumod.vllm-chat.plist'
with plist_path.open('wb') as f:
    plistlib.dump(plist, f)
```

### Jinja2 Approach (Consistent with Existing Architecture)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gpumod.{{ service_id }}</string>
    <key>ProgramArguments</key>
    <array>
        {% for arg in program_arguments %}
        <string>{{ arg }}</string>
        {% endfor %}
    </array>
    <key>KeepAlive</key>
    <{{ 'true' if keep_alive else 'false' }}/>
    ...
</dict>
</plist>
```

---

## 3. launchd vs systemd: Key Differences for gpumod

### Feature Mapping

| systemd Feature | launchd Equivalent | Notes |
|-----------------|-------------------|-------|
| `[Service] ExecStart=` | `ProgramArguments` (array) | launchd requires array, not string |
| `Restart=always` | `KeepAlive=true` | launchd re-launches if process exits |
| `Restart=on-failure` | `KeepAlive.SuccessfulExit=false` | Only restart on non-zero exit |
| `RestartSec=5` | `ThrottleInterval=5` | Min seconds between launches |
| `StartLimitBurst` / `StartLimitIntervalSec` | Built-in: crashes within 10s = stop relaunching | launchd's 10-second rule is hardcoded |
| `Environment=KEY=VAL` | `EnvironmentVariables` dict | Plist dictionary |
| `StandardOutput=journal` | `StandardOutPath=/path` | launchd writes to files, not journal |
| `StandardError=journal` | `StandardErrorPath=/path` | Same |
| `WantedBy=default.target` | `RunAtLoad=true` | Start when agent loads |
| `After=network.target` | No direct equivalent | launchd handles deps differently |
| `Type=notify` | No equivalent | launchd does not support sd_notify |
| Socket activation | `Sockets` dict + `MachServices` | Both support it, different mechanisms |
| `User=` directive | N/A for LaunchAgents | LaunchAgents run as current user |
| `systemctl --user start` | `launchctl load ~/Library/LaunchAgents/...` | Different command |
| `systemctl --user stop` | `launchctl unload ~/Library/LaunchAgents/...` | Different command |
| `journalctl -u service` | `log show --predicate 'process == "name"'` | Apple Unified Logging |

### Critical Differences

1. **No dependency ordering.** launchd does not have `After=` or `Requires=`. Services start independently. gpumod must handle ordering in its own LifecycleManager.

2. **10-second crash rule.** If a process exits within 10 seconds of launch, launchd considers it crashed and may stop relaunching. This affects ML model loading which can take 30+ seconds. Set `ThrottleInterval` appropriately.

3. **No sd_notify.** launchd cannot wait for a service to signal readiness. gpumod already handles this via health endpoint polling, which works the same on macOS.

4. **Logging.** launchd writes stdout/stderr to files (StandardOutPath/StandardErrorPath) rather than a journal. Alternatively, processes can log to Apple's Unified Logging (`os_log`), queryable via `log show`.

5. **Paths.** LaunchAgents go in `~/Library/LaunchAgents/` (user-level, like systemd --user). LaunchDaemons go in `/Library/LaunchDaemons/` (requires root).

6. **File format.** Plist XML vs INI-style systemd units. Both are structured, but plist is more verbose.

7. **Socket activation.** Both support it but launchd sockets are hardcoded into the plist by name, while systemd uses separate .socket units.

### gpumod Architecture Impact

The existing `TemplateEngine` / `UnitFileInstaller` pattern extends naturally:
- Add `LaunchdTemplateEngine` alongside `SystemdTemplateEngine`
- Add `PlistInstaller` alongside `UnitFileInstaller`
- `LifecycleManager` already does health-check-based readiness (no sd_notify dependency)
- Service drivers are OS-independent (they manage HTTP APIs, not systemd units)

The main changes needed:
1. New Jinja2 templates for plist XML in `src/gpumod/templates/launchd/`
2. A `PlistInstaller` that writes to `~/Library/LaunchAgents/`
3. A `LaunchdController` wrapping `launchctl` (analogous to systemctl wrapper)
4. Platform detection to choose systemd vs launchd path

---

## 4. Recommended Implementation Strategy

### Phase 1: GPU Discovery (Minimal Dependencies)

Use **ioreg + plistlib** (stdlib only) for GPU memory metrics:
- `ioreg -r -d 1 -w 0 -c IOAccelerator -a` for GPU usage (system-wide)
- `sysctl hw.memsize` for total RAM
- `sysctl iogpu.wired_limit_mb` for custom GPU limit (or compute default 66-75%)
- `psutil` for system RAM (already likely a dependency)

Optionally add **pyobjc-framework-Metal** for:
- `recommendedMaxWorkingSetSize` (the "correct" GPU capacity number)
- More efficient than subprocess (no ioreg spawn)

### Phase 2: Service Management

Use **plistlib** (stdlib) or **Jinja2** (already a dependency) for plist generation.
Use **subprocess** for `launchctl` commands.
Optionally use **python-launchd** (MIT, v1.0.0) for querying job status.

### Phase 3: Platform Abstraction

Add platform detection in `SystemInfoCollector`:
- Linux: nvidia-smi + /proc/meminfo (existing)
- macOS: ioreg IOAccelerator + sysctl/psutil (new)

Add platform detection in template/installer:
- Linux: systemd units (existing)
- macOS: launchd plists (new)

---

## 5. vLLM on macOS (New Discovery — 2026-03-30)

The epic originally stated "vLLM on macOS" is out of scope because vLLM requires CUDA. **This is no longer true as of early 2026.** Two active projects bring vLLM to Apple Silicon:

### vllm-metal (Official vLLM Project)

- **Repo:** https://github.com/vllm-project/vllm-metal
- **Version:** 0.1.0 (Jan 7, 2026) — Alpha
- **License:** Apache 2.0
- **Python:** 3.12, 3.13
- **What it does:** Community-maintained hardware plugin for vLLM on Apple Silicon. Uses MLX as primary compute backend. Provides MLX-accelerated inference, unified memory with zero-copy operations, experimental paged attention for KV cache.
- **Status:** Alpha — usable but not production-ready.

### vllm-mlx (Third-party)

- **Repo:** https://github.com/waybarrios/vllm-mlx
- **Version:** 0.2.6 (Feb 13, 2026)
- **License:** MIT
- **What it does:** OpenAI and Anthropic-compatible server for Apple Silicon. Integrates mlx-lm, mlx-vlm, mlx-audio, mlx-embeddings. Supports continuous batching, MCP tool calling, multimodal inference (text, images, video, audio). Claims 400+ tok/s.
- **Status:** Actively maintained, more features than vllm-metal.

### Impact on Epic

This changes the "Out of scope" section. vLLM on macOS is now technically feasible via these MLX-backed projects. However, gpumod's VLLMDriver currently wraps vLLM's standard CUDA entrypoint. Supporting vllm-metal or vllm-mlx would require a new driver variant or configuration. **Recommend evaluating in the Phase 1 spike** rather than committing to scope now.

---

## Sources

- [Apple Developer - currentAllocatedSize](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize)
- [Apple Developer - recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- [Apple Developer - Creating Launch Daemons and Agents](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)
- [pyobjc-framework-Metal on PyPI](https://pypi.org/project/pyobjc-framework-Metal/) -- v12.1, MIT, Python >= 3.10
- [MLX Unified Memory docs](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) -- v0.31.1, MIT
- [MLX Metal API](https://ml-explore.github.io/mlx/build/html/python/metal.html)
- [python-launchd on GitHub](https://github.com/infothrill/python-launchd) -- v0.3.0, MIT, dormant since 2021
- [plistlib stdlib docs](https://docs.python.org/3/library/plistlib.html)
- [macmon on GitHub](https://github.com/vladkens/macmon) -- Rust, MIT, sudoless
- [asitop on GitHub](https://github.com/tlkh/asitop) -- Python, MIT, requires sudo, dormant
- [apple-gpu on PyPI](https://pypi.org/project/apple-gpu/) -- v0.3.0, MIT, dormant since Oct 2023
- [vllm-metal on GitHub](https://github.com/vllm-project/vllm-metal) -- v0.1.0, Apache 2.0, Jan 2026
- [vllm-mlx on GitHub](https://github.com/waybarrios/vllm-mlx) -- v0.2.6, MIT, Feb 2026
- [vllm-mlx on PyPI](https://pypi.org/project/vllm-mlx/) -- v0.2.6
- [IOAccelerator PerformanceStatistics](https://forums.macrumors.com/threads/request-share-your-ioreg-ioaccelerator-results-please.2293664/)
- [Apple Silicon GPU memory limits](https://stencel.io/posts/apple-silicon-limitations-with-usage-on-local-llm%20.html)
- [sysctl iogpu.wired_limit_mb](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe)
- [launchd.info](https://www.launchd.info/)
