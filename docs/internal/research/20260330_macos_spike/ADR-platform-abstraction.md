# ADR: Platform Abstraction for macOS Apple Silicon Support

## Status

Proposed

## Date

2026-03-30

## Context

gpumod was built as a Linux-only, NVIDIA-only GPU service manager. Spike
**gpumod-2qc** investigated 16 research questions to determine feasibility
of supporting macOS Apple Silicon (M-series). Researcher agents produced
empirical findings documented in:

- `q2_q3_q4_findings.md` -- GPU memory metrics, pyobjc viability, ioreg semantics
- `q5_q6_findings.md` -- vLLM macOS endpoint compatibility, llama.cpp Metal
- `macos-gpu-memory-launchd.md` -- launchd plist generation, systemd-to-launchd mapping

The spike revealed that macOS support is feasible without disrupting the
existing Linux/NVIDIA path, provided we introduce platform abstractions at
three boundaries: process control (systemd vs launchd), template rendering
(`.service` vs `.plist`), and GPU memory tracking (nvidia-smi vs ioreg/Metal).

This ADR proposes the minimum set of abstractions to support both platforms
while preserving SOLID principles and the existing driver architecture.

---

## Decision

Relax the "Linux only" and "NVIDIA only" constraints by introducing
platform-dispatched abstractions behind Protocols. Existing Linux code
paths remain unchanged; macOS support is added via new implementations
of the same interfaces.

### 1. ProcessController Protocol

Introduce a `ProcessController` Protocol that abstracts the init system
interface. Today, `services/systemd.py` is called directly from
`LifecycleManager` and `UnitFileInstaller`. The new protocol decouples
these callers from any specific init system.

```python
class ProcessController(Protocol):
    async def start(self, service_name: str) -> None: ...
    async def stop(self, service_name: str) -> None: ...
    async def restart(self, service_name: str) -> None: ...
    async def is_active(self, service_name: str) -> bool: ...
    async def get_state(self, service_name: str) -> str: ...
    async def get_logs(self, service_name: str, lines: int = 20) -> list[str]: ...
    def validate_service_name(self, name: str) -> None: ...
```

**Two implementations:**

| Implementation | Wraps | Service location |
|----------------|-------|------------------|
| `SystemdController` | Existing `systemd.py` (`systemctl --user`) | `~/.config/systemd/user/` |
| `LaunchdController` | `launchctl bootstrap/bootout` | `~/Library/LaunchAgents/` |

**Key design points:**

- `_validate_unit_name()` from `systemd.py` (line 85-89) moves into
  `SystemdController.validate_service_name()`. `LaunchdController`
  validates against the `com.gpumod.<name>` label convention instead.
- `LifecycleManager.__init__` accepts a `ProcessController` instead of
  importing `systemd` directly. `ServiceRegistry` passes it through.
- The `_get_systemd_env()` D-Bus helper stays inside `SystemdController`
  -- it is Linux-specific and irrelevant on macOS.
- `LaunchdController` uses `launchctl bootstrap gui/<uid>` and
  `launchctl bootout gui/<uid>` (modern macOS API) rather than the
  deprecated `load`/`unload` commands.
- Log retrieval: `SystemdController` uses `journalctl`;
  `LaunchdController` reads `StandardOutPath`/`StandardErrorPath` files
  or queries Apple Unified Logging via `log show --predicate`.

**Why no sleep/wake in ProcessController:** Q5 findings confirm that
neither vllm-mlx (v0.2.6) nor vllm-metal (v0.1.0) expose sleep/wake
endpoints. Both servers have internal stubs that log "not applicable for
MLX (unified memory)" (vllm-mlx `worker.py:236-239`) and "not supported
on Metal" (vllm-metal `v1/worker.py:355-369`). On Apple Silicon, CPU and
GPU share the same physical memory -- "offloading from GPU to CPU" is a
no-op. Sleep/wake remains a CUDA/ROCm concept; the existing
`ServiceDriver.sleep()`/`wake()` optional methods are sufficient for
drivers that support it. No ProcessController involvement needed.

### 2. Template Engine Platform Dispatch

Extend `_DRIVER_TEMPLATE_MAP` from a 1D `dict[str, str]` to a 2D
`dict[str, dict[str, str]]` keyed by `[driver][platform]`:

```python
_DRIVER_TEMPLATE_MAP: dict[str, dict[str, str]] = {
    "vllm":     {"linux": "vllm.service.j2",     "darwin": "vllm.plist.j2"},
    "llamacpp": {"linux": "llamacpp.service.j2",  "darwin": "llamacpp.plist.j2"},
    "fastapi":  {"linux": "fastapi.service.j2",   "darwin": "fastapi.plist.j2"},
}
```

**Platform auto-detection** uses `sys.platform` (returns `"linux"` or
`"darwin"`), injectable for testing.

**Template directory layout:**

```
src/gpumod/templates/
  systemd/           # existing -- unchanged
    vllm.service.j2
    llamacpp.service.j2
    fastapi.service.j2
  launchd/           # new
    vllm.plist.j2
    llamacpp.plist.j2
    fastapi.plist.j2
```

Each platform subdirectory gets its own `FileSystemLoader`. The existing
`_SAFE_NAME_RE` pattern (`^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_.]+$`) on line 27
of `engine.py` is already sufficient for `.plist.j2` filenames -- no change
needed.

**Jinja2 SandboxedEnvironment** already handles XML output correctly.
Plist templates use the same `SandboxedEnvironment` + `StrictUndefined`
configuration as systemd templates. The launchd research document
(section 2) confirms that Jinja2 can render plist XML with the same
patterns used for systemd INI-style units.

**`UnitFileInstaller` becomes `ServiceFileInstaller`** (renamed for
platform neutrality). On Linux it writes to `~/.config/systemd/user/`
and calls `systemctl --user daemon-reload`. On macOS it writes to
`~/Library/LaunchAgents/` -- no daemon-reload equivalent is needed
(launchd reads plists at `bootstrap` time).

### 3. GPU Memory Abstraction

Introduce a `GPUMemoryTracker` Protocol:

```python
class GPUMemoryTracker(Protocol):
    async def get_capacity_mb(self) -> int: ...
    async def get_used_mb(self) -> int: ...
    async def get_free_mb(self) -> int: ...
    async def wait_for_release(
        self, required_mb: int, timeout_s: float = 60.0
    ) -> bool: ...
```

**Two implementations:**

| Implementation | Data source | Capacity source |
|----------------|-------------|-----------------|
| `NvidiaGPUTracker` | Existing `VRAMTracker` (nvidia-smi / pynvml) | `nvidia-smi --query-gpu=memory.total` |
| `MetalGPUTracker` | `ioreg IOAccelerator` "Alloc system memory" via `plistlib` | `recommendedMaxWorkingSetSize` via pyobjc Metal |

**Critical empirical findings (Q2-Q4) that inform the design:**

1. **Capacity: 78%, not 66%.** On the test machine (M2 Max, 32 GB, macOS
   26.4), `recommendedMaxWorkingSetSize` returned 25,559 MB -- 78% of
   total RAM. Community sources citing 66% or 75% are outdated
   (2022-era macOS). The implementation must always query the Metal API
   at runtime; hardcoding any percentage is wrong.

2. **Usage: "Alloc system memory", not "In use system memory".** Q4
   testing showed "In use system memory" dropped from 469 MB to 277 MB
   while a model was still loaded and idle. It tracks in-flight GPU
   operations, not total allocations. "Alloc system memory" increased
   monotonically with model load (+174 MB for the test model) and
   decreased after server exit. This is the correct metric.

3. **Release speed: < 3 seconds** for the toy model (stories260K).
   After llama-server exit, "Alloc system memory" dropped within 3
   seconds. For production models (7B+), the release time is unknown and
   must be polled, matching the existing `wait_for_vram_release()` pattern
   in `VRAMTracker`.

4. **pyobjc dependency is safe.** pyobjc-framework-Metal v12.1: pre-built
   wheels for Python 3.10-3.14 on macOS universal2, 5 ms install, ~5 MB
   total, MIT license, actively maintained. No Xcode CLI tools required
   for wheel install.

5. **`iogpu.wired_limit_mb` sysctl** can override GPU memory limits
   (requires sudo). `MetalGPUTracker` should check this value and warn
   if non-zero, since it may cause `recommendedMaxWorkingSetSize` to
   report a lower limit than the Metal default.

### 4. RAM Check Portability

`RAMCheck` (in `preflight/ram_check.py`) currently reads `/proc/meminfo`
directly. On macOS, `/proc/meminfo` does not exist.

**macOS replacement:** `sysctl hw.memsize` returns total physical RAM
(verified: 34,359,738,368 bytes = 32 GB on test machine). `vm_stat`
provides page-level memory breakdown (free, active, inactive, speculative,
wired). Available memory can be computed from `vm_stat` output.

**Implementation approach:** Introduce a `RAMInfoProvider` Protocol with
`LinuxRAMInfo` (reads `/proc/meminfo`) and `DarwinRAMInfo` (reads
`sysctl hw.memsize` + `vm_stat`). `RAMCheck.__init__` accepts the
provider instead of a `meminfo_path`. The existing `meminfo_path`
injection point on line 43 of `ram_check.py` shows the testability
pattern already exists -- we are generalizing it.

### 5. VRAM Management Strategy (macOS)

**On Apple Silicon, there is no separate VRAM.** CPU and GPU share
unified memory. The vLLM sleep/wake mechanism (L1: offload weights from
GPU VRAM to CPU RAM via PCIe; L2: discard weights entirely) is
architecturally impossible -- there is no PCIe bus and no separate
memory pools.

**Empirical confirmation (Q5):**
- vllm-mlx `platform.py:159`: `is_sleep_mode_available()` returns `False`
- vllm-mlx `worker.py:236-239`: `sleep()` logs "not applicable for MLX"
- vllm-metal `v1/worker.py:355-369`: `sleep()` logs "not supported on Metal"
- Both servers return HTTP 404 for `/sleep`, `/wake_up`, `/is_sleeping`

**macOS VRAM management strategy:** Use `launchd bootout/bootstrap` (stop
and start the service process) as the sole memory release mechanism. When
a mode switch requires freeing GPU memory, gpumod stops the current
service via `LaunchdController.stop()` and polls `MetalGPUTracker` until
"Alloc system memory" drops below the required threshold.

**Release timing (Q4):** Memory release was < 3 seconds for the toy model
(stories260K, 1.1 MB). For production-scale models, `wait_for_release()`
should use the same polling pattern as the existing `VRAMTracker` (0.5s
poll interval, 60s timeout).

The existing `LlamaCppDriver` Router sleep level (model unload/reload
via `/slots`) remains an option on macOS for llama.cpp services
specifically, but the primary strategy is stop/start.

### 6. DockerDriver

The existing `DockerDriver` is already platform-independent -- it wraps
`docker` CLI commands that work identically on Linux and macOS.

**No code changes needed.**

**Caveat:** GPU passthrough is not available on Docker Desktop for macOS.
Docker containers cannot access Apple Silicon GPU acceleration. This is
a Docker Desktop limitation, not a gpumod issue. The `DockerDriver`
continues to manage container lifecycle on macOS, but GPU-accelerated
inference inside Docker containers is not supported on Apple Silicon.

### 7. Preferred ML Server on macOS

**Recommendation: vllm-mlx (v0.2.6) over vllm-metal (v0.1.0).**

Empirical comparison from Q5 testing (model: mlx-community/Qwen2.5-0.5B-Instruct-4bit):

| Criterion | vllm-mlx | vllm-metal |
|-----------|----------|------------|
| Critical endpoints (4/4) | All pass | All pass |
| Warm startup time | **3.2s** | 14.5s (4.5x slower) |
| `/health` response time | 1.6ms | 0.9ms |
| Total HTTP endpoints | 15 | 4 |
| License | MIT | Apache 2.0 |
| Maintainer | Third-party (waybarrios) | Official vLLM community |
| Additional features | Embeddings, MCP, audio, vision | Core OpenAI-compat only |

**vllm-mlx is preferred because:**
1. 4.5x faster warm startup (3.2s vs 14.5s) -- compatible with launchd ThrottleInterval=10
2. Richer monitoring via `/v1/status` endpoint (Metal memory info, request details)
3. More actively developed with broader feature set

**Risks:**
- Both are sub-v1.0 and experimental
- vllm-mlx is third-party, not under the vllm-project GitHub org
- vllm-mlx has a version mismatch: PyPI 0.2.6 but `__version__` reports 0.2.5
- vllm-mlx pulls more dependencies (101 packages including gradio, mcp, audio)

**Fallback:** llama.cpp (Homebrew build 3614, commit a1631e53) is verified
present and functional on the test machine. llama-server exposes the same
HTTP API on Metal as on CUDA -- the `LlamaCppDriver` requires no API
changes. llama.cpp is the conservative fallback if vllm-mlx proves
unreliable.

---

## Options Considered

### Option A: Platform Factory Above Engine (Rejected)

A `PlatformFactory` class that selects the correct template engine,
process controller, and GPU tracker based on `sys.platform` and returns
a pre-configured bundle.

**Rejected because:**
- Violates **Single Responsibility** -- the factory accumulates knowledge
  of every platform-specific component.
- Splits template logic: `TemplateEngine` would no longer own its own
  template selection; the factory would choose templates and the engine
  would render them. This breaks the current cohesion where
  `render_service_unit()` selects the template from the driver map and
  renders it in one call.
- Makes testing harder -- callers must mock the factory rather than
  injecting individual components.

### Option B: Platform Dispatch in Engine + Protocol Injection (Chosen)

Add a platform dimension to `_DRIVER_TEMPLATE_MAP` inside `TemplateEngine`
and inject `ProcessController` / `GPUMemoryTracker` via constructor
parameters into `LifecycleManager`, `ServiceFileInstaller`, and
`VRAMCheck`.

**Chosen because:**
- **Single Responsibility**: Each abstraction owns one concern.
  `ProcessController` owns init system interaction. `GPUMemoryTracker`
  owns GPU memory queries. `TemplateEngine` owns template selection and
  rendering.
- **Open/Closed**: macOS support is added by implementing new Protocol
  classes (`LaunchdController`, `MetalGPUTracker`, `DarwinRAMInfo`),
  not by modifying `SystemdController`, `NvidiaGPUTracker`, or
  `LinuxRAMInfo`.
- **Liskov Substitution**: `LaunchdController` and `SystemdController`
  are interchangeable behind `ProcessController`. Tests can inject mocks
  for either.
- **Interface Segregation**: Protocols are small and focused --
  `ProcessController` has 7 methods, `GPUMemoryTracker` has 4.
- **Dependency Inversion**: `LifecycleManager` depends on
  `ProcessController` (abstraction), not `systemd.py` (implementation).

### Option C: Separate macOS Module Tree (Rejected)

Duplicate the `services/` tree as `services_macos/` with macOS-specific
implementations of every module.

**Rejected because:**
- Violates **DRY** -- service drivers (`VLLMDriver`, `LlamaCppDriver`,
  `FastAPIDriver`) are platform-independent (they make HTTP calls). Only
  the process control, template, and GPU tracking layers differ.
- Maintenance burden: every bug fix or feature in the shared driver logic
  must be applied to both trees.
- The actual platform-specific surface is small (~7 new files vs ~18
  existing files in `services/`).

---

## Consequences

### Positive

- **Existing Linux/NVIDIA path is unchanged.** `SystemdController`
  wraps the existing `systemd.py` code. `NvidiaGPUTracker` wraps the
  existing `VRAMTracker`. No regressions. (Open/Closed Principle)
- **Service drivers become platform-agnostic.** `VLLMDriver`,
  `LlamaCppDriver`, `FastAPIDriver`, and `DockerDriver` make HTTP calls
  and do not touch the init system. The `ProcessController` injection
  means drivers work on any platform without modification. (Dependency
  Inversion Principle)
- **Testable via Protocol injection.** Unit tests inject mock
  `ProcessController` and `GPUMemoryTracker` instances -- no real systemd
  or launchd needed. Integration tests can use the real controllers.
  (Interface Segregation Principle)
- **Template rendering stays cohesive.** `TemplateEngine` still owns
  template selection and rendering; the platform dimension is just an
  additional key in the existing driver map. (Single Responsibility Principle)

### Negative

- **~18 files modified, ~7 new files.** The Protocol definitions,
  macOS implementations, launchd templates, and wiring changes touch a
  significant portion of the services layer.
- **Sub-v1.0 dependency on vllm-mlx.** The recommended ML server on
  macOS is third-party and experimental. API breakage is possible between
  minor versions.
- **pyobjc dependency (macOS only).** Adds pyobjc-framework-Metal
  (~5 MB) as an optional dependency. Required for querying
  `recommendedMaxWorkingSetSize` accurately -- the alternative (hardcoding
  a percentage) is provably wrong per Q2 findings (78% vs assumed 66%).
- **Undocumented ioreg keys.** IOAccelerator `PerformanceStatistics`
  fields like "Alloc system memory" are not part of a public Apple API.
  They may change across macOS versions.

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| vllm-mlx abandonment | Medium | High | Fallback to llama.cpp (verified functional, same HTTP API). LlamaCppDriver needs no changes. |
| macOS Metal API changes | Low | Medium | pyobjc tracks Xcode releases. `recommendedMaxWorkingSetSize` has been stable since Metal 1.0. |
| 78% memory threshold varies by macOS version | Medium | Low | Always query `recommendedMaxWorkingSetSize` at runtime; never hardcode. This is the design. |
| IOAccelerator PerformanceStatistics key rename | Low | Medium | Wrap ioreg parsing with graceful fallback. Log warning if expected keys are absent. |
| launchd 10-second crash rule kills slow model loads | Low | Low | The 10-second rule applies to exit-within-10-seconds-of-launch. vllm-mlx warm start is 3.2s but that is time-to-healthy, not time-to-exit. Model loading takes 30+ seconds; the process does not exit during load. ThrottleInterval handles restart pacing. |
| Docker GPU passthrough on macOS | N/A | Low | Document as not supported. DockerDriver still manages container lifecycle for CPU-only workloads. |

---

## Follow-up Tickets

Implementation should proceed in this order (respecting dependency chain):

1. **Protocol definitions** -- Define `ProcessController`,
   `GPUMemoryTracker`, and `RAMInfoProvider` Protocols in
   `src/gpumod/services/protocols.py` (new file).

2. **SystemdController** -- Extract existing `systemd.py` functions into
   a `SystemdController` class implementing `ProcessController`. Existing
   callers updated to use the Protocol.

3. **LaunchdController** -- Implement `ProcessController` for macOS
   using `launchctl bootstrap/bootout gui/<uid>`. Include service name
   validation (`com.gpumod.<name>` convention).

4. **MetalGPUTracker** -- Implement `GPUMemoryTracker` using
   `ioreg + plistlib` for current usage and `pyobjc Metal` for capacity
   (`recommendedMaxWorkingSetSize`).

5. **NvidiaGPUTracker** -- Wrap existing `VRAMTracker` behind the
   `GPUMemoryTracker` Protocol (adapter pattern, minimal changes).

6. **Template engine platform dispatch** -- Add 2D
   `_DRIVER_TEMPLATE_MAP`, platform-specific `FileSystemLoader`, and
   `render_service_unit()` platform parameter.

7. **Launchd Jinja2 templates** -- Create `vllm.plist.j2`,
   `llamacpp.plist.j2`, `fastapi.plist.j2` in
   `src/gpumod/templates/launchd/`.

8. **ServiceFileInstaller** -- Rename `UnitFileInstaller`, add
   platform-dispatched write path (`~/Library/LaunchAgents/` on macOS).

9. **RAMInfoProvider** -- Implement `DarwinRAMInfo` using
   `sysctl hw.memsize` + `vm_stat`. Adapt `RAMCheck` to use the Protocol.

10. **LifecycleManager wiring** -- Inject `ProcessController` into
    `LifecycleManager`. Replace direct `systemd.get_unit_state()` and
    `systemd.journal_logs()` calls with Protocol methods.

11. **VRAMCheck + preflight wiring** -- Update `VRAMCheck` to accept
    `GPUMemoryTracker` instead of `VRAMTracker` directly.

12. **Documentation** -- Update `docs/architecture/index.md` with
    platform abstraction layer, `docs/getting-started/cli.md` with
    macOS-specific notes.
