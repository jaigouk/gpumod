---
title: Architecture Overview
description: System architecture for gpumod — a GPU service manager for ML workloads on single-GPU Linux systems.
---

# gpumod Architecture

## 1. Introduction and Goals

### What is gpumod?

**gpumod** orchestrates ML inference services (vLLM, llama.cpp, FastAPI, Docker) on single-GPU Linux systems. It solves the problem of efficiently sharing limited GPU VRAM across multiple services that cannot all run simultaneously.

### Key Requirements

| Requirement           | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| **VRAM Safety**       | Never start services that would exceed GPU capacity           |
| **Fast Switching**    | Switch between service configurations in seconds, not minutes |
| **Zero Downtime**     | Simulate before deploy — catch OOM before it happens          |
| **Unified Interface** | Single abstraction for vLLM, llama.cpp, FastAPI, Docker       |
| **AI Integration**    | Expose GPU management to AI assistants via MCP                |

### Stakeholders

| Role              | Concerns                                              |
| ----------------- | ----------------------------------------------------- |
| **ML Engineers**  | Quick mode switching, VRAM visibility, service health |
| **AI Assistants** | Programmatic GPU control via MCP tools                |
| **DevOps**        | Systemd integration, monitoring, automation           |

---

## 2. Constraints

### Technical Constraints

| Constraint       | Rationale                                                 |
| ---------------- | --------------------------------------------------------- |
| **Linux only**   | Requires systemd for service management                   |
| **Single GPU**   | Multi-GPU adds complexity; most local setups have one GPU |
| **NVIDIA only**  | Relies on nvidia-smi for VRAM queries                     |
| **Python 3.11+** | Modern async, type hints, pattern matching                |

### Organizational Constraints

| Constraint                | Rationale                                    |
| ------------------------- | -------------------------------------------- |
| **No root required**      | Uses `systemctl --user` for service control  |
| **Offline-capable**       | Core functionality works without network     |
| **Configuration as code** | YAML presets, Jinja2 templates, SQLite state |

---

## 3. Context and Scope

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Systems                         │
├─────────────────────────────────────────────────────────────────┤
│  AI Assistants        systemd           nvidia-smi              │
│  (Claude Code,        (service          (GPU queries)           │
│   Cursor, etc.)       lifecycle)                                │
└────────┬───────────────────┬─────────────────────┬──────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                          gpumod                                 │
│                                                                 │
│  Orchestrates GPU services, tracks VRAM, manages modes          │
│                                                                 │
└────────┬───────────────────┬─────────────────────┬──────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ML Services                               │
├─────────────────────────────────────────────────────────────────┤
│  vLLM servers      llama.cpp servers      FastAPI/Docker        │
│  (embedding,       (code completion,      (ASR, TTS,            │
│   chat, hyde)       reasoning)             custom)              │
└─────────────────────────────────────────────────────────────────┘
```

### Scope

**In Scope:**

- Service lifecycle (start, stop, health check)
- VRAM tracking and simulation
- Mode-based service grouping
- Sleep/wake for VRAM optimization
- MCP server for AI integration

**Out of Scope:**

- Model training
- Model serving (delegated to vLLM/llama.cpp)
- Multi-node orchestration (use Kubernetes)

---

## 4. Solution Strategy

### Core Decisions

| Decision                | Approach                        | Rationale                                                        |
| ----------------------- | ------------------------------- | ---------------------------------------------------------------- |
| **Service Abstraction** | Driver pattern                  | Hide vLLM/llama.cpp/FastAPI differences behind unified interface |
| **State Management**    | SQLite                          | Embedded, transactional, zero-setup, portable                    |
| **Configuration**       | YAML presets + Jinja2 templates | Human-readable, version-controllable, flexible                   |
| **Process Control**     | systemd user services           | Standard Linux, no daemon, survives crashes                      |
| **VRAM Tracking**       | nvidia-smi polling              | Universal NVIDIA support, no driver dependencies                 |
| **AI Integration**      | MCP protocol                    | Standard interface for Claude, Cursor, etc.                      |

### Key Quality Goals

| Goal                | Approach                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| **Reliability**     | Pre-flight VRAM simulation prevents OOM; health checks detect failures |
| **Performance**     | Sleep states preserve VRAM; wake time <1s for L1, <2s for L2           |
| **Maintainability** | Driver pattern isolates runtime-specific code; 1300+ unit tests        |
| **Usability**       | Rich CLI, interactive TUI, AI-friendly MCP tools                       |

---

## 5. Building Block View

### Level 1: System Decomposition

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│           CLI · Interactive TUI · MCP Server                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   SERVICE LAYER                              │
├──────────────────────────────────────────────────────────────┤
│  ServiceManager    ← Orchestrates all operations             │
│  ├── ServiceRegistry    ← Discovers & tracks services        │
│  ├── LifecycleManager   ← Start/stop with dependencies       │
│  ├── HealthMonitor      ← Continuous health checking         │
│  ├── VRAMTracker        ← GPU memory tracking                │
│  └── SleepController    ← Sleep/wake management              │
│                                                              │
│  Service Drivers:                                            │
│  ├── VLLMDriver         ← vLLM serve processes               │
│  ├── LlamaCppDriver     ← llama.cpp server                   │
│  ├── FastAPIDriver      ← Custom FastAPI servers             │
│  └── DockerDriver       ← Containerized services             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONFIGURATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  SQLite DB: services, modes, settings                       │
│  Templates: Jinja2 for systemd units                        │
│  Presets: YAML service definitions                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   SYSTEM LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  systemd: unit management                                   │
│  nvidia-smi: GPU info, VRAM usage                           │
│  HTTP: health endpoints, sleep/wake APIs                    │
└─────────────────────────────────────────────────────────────┘
```

### Level 2: Discovery Layer

The discovery layer provides read-only access to external knowledge sources
(HuggingFace, GitHub) without LLM calls. These components back the Tier 2
MCP tools.

```
┌──────────────────────────────────────────────────────────────┐
│                     DISCOVERY LAYER                          │
├──────────────────────────────────────────────────────────────┤
│  HuggingFaceSearcher   ← Search HF models by query           │
│  GGUFMetadataFetcher   ← GGUF file listing + VRAM estimates  │
│  ConfigFetcher         ← Model config.json (architecture)    │
│  DriverDocsFetcher     ← Official llama.cpp/vLLM docs        │
│  PresetGenerator       ← YAML service configs                │
│  SystemInfoCollector   ← GPU, RAM, driver info               │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:** MCP tools -> Discovery components -> HuggingFace API / GitHub raw

### Level 2: Consulting Layer

The consulting layer provides multi-step reasoning for complex queries using
RLM (Recursive Language Models). It is optional and read-only — returns
recommendations but never executes mutating operations.

```
┌──────────────────────────────────────────────────────────────┐
│                    CONSULTING LAYER                          │
├──────────────────────────────────────────────────────────────┤
│  GpumodConsultEnv      ← Custom RLM environment              │
│  ├── Whitelisted read-only MCP tools                         │
│  ├── AST-validated code execution                            │
│  └── Timeout enforcement (5s per block, 60s total)           │
│  RLMOrchestrator       ← Manages RLM lifecycle               │
│  └── ConsultResult     ← Structured recommendation output    │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:** `consult` MCP tool -> RLMOrchestrator -> GpumodConsultEnv ->
whitelisted read-only tools (gpu_status, list_gguf_files, etc.)

### Level 2: Service Layer Components

#### ServiceManager

The top-level orchestrator coordinating mode switches and status queries.

**Responsibilities:**

- Validate mode switches against VRAM capacity
- Coordinate service start/stop ordering
- Detect and clean up orphan services
- Wait for VRAM release between transitions

#### ServiceRegistry

Maintains the catalog of registered services and their drivers.

**Responsibilities:**

- Load services from database
- Match services to appropriate drivers
- Track running/sleeping service state

#### LifecycleManager

Handles individual service lifecycle with dependency ordering.

**Responsibilities:**

- Start services in topological order (dependencies first)
- Stop services in reverse order (dependents first)
- Wait for health endpoints after start
- Tail logs on startup failures

#### VRAMTracker

Monitors GPU memory and estimates service requirements.

**Responsibilities:**

- Query nvidia-smi for current usage
- Estimate VRAM for stopped services
- Wait for VRAM release after service stops
- Calculate KV cache requirements

#### SleepController

Manages GPU memory optimization via sleep states.

**Responsibilities:**

- Send sleep/wake commands to drivers
- Track sleep state per service
- Handle wake-on-demand for sleeping services

#### HealthMonitor

Provides continuous health checking for running services.

**Responsibilities:**

- Poll health endpoints at configurable intervals
- Debounce state transitions
- Apply exponential backoff on failures

### Level 3: Service Drivers

Drivers implement the `ServiceDriver` ABC, abstracting runtime differences:

| Driver             | Process Control | Sleep Support              | Health Check |
| ------------------ | --------------- | -------------------------- | ------------ |
| **VLLMDriver**     | systemd         | L1, L2 (via API)           | `/health`    |
| **LlamaCppDriver** | systemd         | Router (model load/unload) | `/health`    |
| **FastAPIDriver**  | systemd         | Custom (if implemented)    | Configurable |
| **DockerDriver**   | Docker API      | Container stop/start       | Configurable |

---

## 6. Runtime View

### Mode Switch Sequence

The most important runtime scenario — switching from one GPU configuration to another.

```
User                    ServiceManager          LifecycleManager        VRAMTracker
  │                           │                        │                     │
  │  switch_mode("rag")       │                        │                     │
  │──────────────────────────>│                        │                     │
  │                           │                        │                     │
  │                           │  validate mode exists  │                     │
  │                           │─────────────┐          │                     │
  │                           │             │          │                     │
  │                           │<────────────┘          │                     │
  │                           │                        │                     │
  │                           │  get running services  │                     │
  │                           │────────────────────────────────────────────> │
  │                           │                        │                     │
  │                           │  compute to_stop, to_start, orphans          │
  │                           │─────────────┐          │                     │
  │                           │             │          │                     │
  │                           │<────────────┘          │                     │
  │                           │                        │                     │
  │                           │  check VRAM capacity   │                     │
  │                           │────────────────────────────────────────────> │
  │                           │                        │                     │
  │                           │  stop outgoing         │                     │
  │                           │───────────────────────>│                     │
  │                           │                        │  systemctl stop     │
  │                           │                        │                     │
  │                           │  wait for VRAM release │                     │
  │                           │────────────────────────────────────────────> │
  │                           │                        │  poll nvidia-smi    │
  │                           │                        │                     │
  │                           │  start incoming        │                     │
  │                           │───────────────────────>│                     │
  │                           │                        │  systemctl start    │
  │                           │                        │  wait for health    │
  │                           │                        │                     │
  │  ModeResult(success=True) │                        │                     │
  │<──────────────────────────│                        │                     │
```

### Service State Machine

```
                    ┌─────────┐
                    │ STOPPED │
                    └────┬────┘
                         │ start()
                         ▼
                    ┌─────────┐
                    │STARTING │
                    └────┬────┘
                         │ health OK
                         ▼
    ┌───────────────┬─────────┬───────────────┐
    │               │ RUNNING │               │
    │               └────┬────┘               │
    │                    │                    │
    │ sleep()            │ stop()             │ health fail
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐         ┌───────────┐
│SLEEPING │        │STOPPING │         │ UNHEALTHY │
└────┬────┘        └────┬────┘         └─────┬─────┘
     │                  │                    │
     │ wake()           │                    │ recover
     │                  ▼                    │
     │             ┌─────────┐               │
     └────────────>│ STOPPED │<──────────────┘
                   └─────────┘
```

---

## 7. Deployment View

### Single-User Deployment

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Machine                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ~/.config/gpumod/                                               │
│  ├── gpumod.db              ← SQLite state                       │
│  └── systemd/user/          ← Generated unit files               │
│      ├── gpumod-vllm-chat.service                                │
│      ├── gpumod-glm-code.service                                 │
│      └── ...                                                     │
│                                                                  │
│  User processes:                                                 │
│  ├── gpumod CLI/TUI         ← Interactive user                   │
│  ├── gpumod MCP server      ← AI assistant integration           │
│  └── systemd user services  ← ML inference processes             │
│                                                                  │
│  System:                                                         │
│  ├── nvidia-smi             ← GPU queries                        │
│  └── systemd --user         ← Service lifecycle                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### MCP Integration

gpumod exposes 16 tools and 8 resources via MCP, organized into three tiers:

```
┌──────────────────────────────────────────────────────────────────┐
│                   AI Assistant (Claude Code)                     │
└───────────────────────────────┬──────────────────────────────────┘
                                │ stdio (JSON-RPC)
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      gpumod MCP Server                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tier 1 - Direct (0 LLM calls):                                  │
│    gpu_status, list_services, list_modes, service_info,          │
│    model_info, simulate_mode, switch_mode, start_service,        │
│    stop_service                                                  │
│                                                                  │
│  Tier 2 - Discovery (0 LLM calls):                               │
│    search_hf_models, list_gguf_files, list_model_files,          │
│    fetch_model_config, generate_preset, fetch_driver_docs        │
│                                                                  │
│  Tier 3 - Consulting (N LLM calls):                              │
│    consult (RLM-based multi-step reasoning)                      │
│                                                                  │
│  Resources:                                                      │
│    gpumod://modes, gpumod://services, gpumod://models,           │
│    gpumod://help, ...                                            │
└───────────────────────────────┬──────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ ServiceManager │   │ Discovery Layer  │   │ Consulting Layer│
│ (Tier 1 ops)   │   │ (Tier 2 fetches) │   │ (Tier 3 RLM)    │
└────────────────┘   └──────────────────┘   └─────────────────┘
```

#### Tool Tiers

| Tier | Tools | LLM Calls | Use Case |
| ---- | ----- | --------- | -------- |
| **1 - Direct** | gpu_status, list_services, list_modes, service_info, model_info, simulate_mode, switch_mode, start_service, stop_service | 0 | Routine operations |
| **2 - Discovery** | search_hf_models, list_gguf_files, list_model_files, fetch_model_config, generate_preset, fetch_driver_docs | 0 | External data lookup |
| **3 - Consulting** | consult | N | Complex multi-step reasoning |

#### RLM Consult Tool

The `consult` tool uses RLM to answer questions like "Can I run Qwen3-235B on
24GB?" by programmatically exploring model metadata, driver docs, and VRAM
constraints across multiple turns.

```
┌──────────────────────────────────────────────────────────────────┐
│                  consult MCP tool                                │
├──────────────────────────────────────────────────────────────────┤
│  RLMOrchestrator                                                 │
│  └── GpumodConsultEnv (NonIsolatedEnv)                           │
│      │                                                           │
│      │  Whitelisted (read-only):        Blocked (NameError):     │
│      │    gpu_status                      switch_mode            │
│      │    list_gguf_files                 start_service          │
│      │    fetch_model_config              stop_service           │
│      │    fetch_driver_docs                                      │
│      │    search_hf_models                                       │
│      │    simulate_mode                                          │
│      │    generate_preset                                        │
│      │    json, re (stdlib)                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Cross-cutting Concepts

### VRAM Management

GPU VRAM is the central constraint. gpumod addresses this through:

1. **Pre-flight simulation** — Calculate VRAM before starting services
2. **Wait-for-release** — Poll nvidia-smi after stops until VRAM freed
3. **Sleep states** — L1/L2/Router reduce VRAM while keeping services warm
4. **Orphan detection** — Clean up services not in current mode definition

#### VRAM Protection Mechanism (Defense-in-Depth)

The system crash risk from VRAM exhaustion is mitigated at multiple layers:

```
Mode Switch Flow with VRAM Protection
======================================

User: gpumod mode switch rag
         │
         ▼
┌────────────────────────────────────┐
│  1. Pre-flight VRAM Simulation     │ ◄── Layer 1: Block if target mode > total VRAM
│     (ServiceManager.switch_mode)   │
└────────────────┬───────────────────┘
                 │ ✓ Total VRAM fits
                 ▼
┌────────────────────────────────────┐
│  2. Stop Outgoing Services         │
│     (LifecycleManager.stop)        │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│  3. Wait for VRAM Release          │ ◄── Layer 2: Timeout = FATAL ERROR (not warning)
│     (VRAMTracker.wait_for_vram)    │     Prevents: CUDA async release race condition
│                                    │     On timeout: Returns ModeResult(success=False)
└────────────────┬───────────────────┘
                 │ ✓ VRAM released
                 ▼
┌────────────────────────────────────┐
│  4. Start Incoming Services        │
│     (LifecycleManager.start)       │
│         │                          │
│         └──► wake() ───────────────│◄── Layer 3: Driver VRAM preflight
│             (LlamaCppDriver)       │     Checks: free_mb >= vram_mb + 512MB margin
│                                    │     Raises: InsufficientVRAMError if insufficient
└────────────────────────────────────┘
```

**Race Condition Explanation:**

CUDA memory release is asynchronous. When `systemctl stop` returns, the GPU memory
may still be held by the kernel driver for milliseconds to seconds. Without waiting,
a subsequent model load can see "30GB needed vs 714MB free" and crash the system.

**Layer Details:**

| Layer | Component | Check | Failure Mode |
| ----- | --------- | ----- | ------------ |
| **1** | ServiceManager pre-flight | Target mode total < GPU capacity | `ModeResult(success=False)` |
| **2** | VRAM wait timeout | free_mb >= required_mb within 120s | `ModeResult(success=False)` |
| **3** | Driver preflight (llama.cpp) | free_mb >= service.vram_mb + 512MB | `InsufficientVRAMError` |

**Manual curl Attack Vector:**

Direct `curl POST /models/load` bypasses gpumod's ServiceManager. Mitigation:
- Layer 3 (driver preflight) catches this when `VRAMTracker` is injected
- llama-router services bind to `127.0.0.1` (configurable), reducing external attack surface
- Future: HTTP proxy layer for comprehensive protection (tracked in backlog)

### Sleep Levels

| Level      | Wake Time | VRAM Saved | Mechanism                     |
| ---------- | --------- | ---------- | ----------------------------- |
| **L1**     | ~9ms      | Partial    | CUDA context preserved        |
| **L2**     | ~1-2s     | Full       | Everything discarded          |
| **Router** | ~500ms    | Full       | Model unloaded from llama.cpp |

### Health Checking

All services expose HTTP health endpoints. gpumod:

- Polls at configurable intervals
- Requires consecutive failures before marking unhealthy
- Uses exponential backoff with jitter
- Tails journalctl logs on startup timeout

### Knowledge Sources

The discovery and consulting layers draw from a 4-tier source hierarchy.
See `research/knowledge_sources.md` in the repository for the full catalog.

| Tier | Source Type | TTL | Examples |
| ---- | ----------- | --- | -------- |
| **1 - Core** | Official tool docs | 24h | llama.cpp server README, vLLM engine args |
| **2 - Curated** | Trusted guides | 7 days | MoE offload guide, Unsloth docs |
| **3 - Dynamic** | API data | 1h | HF config.json, GGUF file listings |
| **4 - Community** | Manual curation | Manual | VRAM rules, model quirks |

Caching uses TTL-based expiry per tier. Tier 1-2 sources fall back to cached
copies on fetch failure. Tier 3 data is always fresh from APIs.

### Function Whitelisting

RLM code execution uses function whitelisting as the primary security control.
Only explicitly exposed functions are callable from LLM-generated code.

**Whitelisted (read-only):**
- `gpu_status`, `list_gguf_files`, `fetch_model_config`, `fetch_driver_docs`,
  `search_hf_models`, `simulate_mode`, `generate_preset`
- Standard library: `json`, `re`

**Blocked (raise `NameError`):**
- `switch_mode`, `start_service`, `stop_service` (mutating operations)
- `import`, `exec`, `eval`, `open` (blocked via AST validation)

**AST validation:** Before executing LLM-generated code, an AST check rejects
`import` statements, `exec`/`eval` calls, and file I/O. This provides defense
in depth alongside the restricted namespace.

**Timeouts:** 5 seconds per code block, 60 seconds total per consult call.

### Security

| Area            | Control                                                               |
| --------------- | --------------------------------------------------------------------- |
| **Docker**      | Privileged mode blocked, host network blocked, unsafe mounts rejected |
| **TUI**         | Untrusted text sanitized to prevent markup injection                  |
| **AI Planning** | LLM suggestions are advisory only, never auto-executed                |
| **RLM Consult** | Function whitelisting, AST validation, read-only tools only           |

---

## 9. Architecture Decisions

### ADR-1: SQLite for State

**Context:** Need persistent storage for services, modes, settings.

**Decision:** Embedded SQLite database.

**Consequences:**

- (+) Zero setup, no daemon
- (+) ACID guarantees for mode switches
- (+) Single-file backup
- (-) No concurrent write access (acceptable for single-user)

### ADR-2: Driver Pattern for Services

**Context:** vLLM, llama.cpp, FastAPI, Docker have different APIs.

**Decision:** Abstract behind `ServiceDriver` interface.

**Consequences:**

- (+) Unified start/stop/health interface
- (+) Easy to add new runtimes
- (+) Mockable for testing
- (-) Some runtime-specific features harder to expose

### ADR-3: systemd User Services

**Context:** Need process supervision without root.

**Decision:** Use `systemctl --user` for service lifecycle.

**Consequences:**

- (+) Standard Linux, well-documented
- (+) Automatic restart on crash
- (+) No root required
- (-) Linux-only (no macOS/Windows)

### ADR-4: VRAM Simulation Before Deploy

**Context:** OOM crashes disrupt running services.

**Decision:** Simulate VRAM requirements before `systemctl start`.

**Consequences:**

- (+) Zero-downtime experimentation
- (+) Suggests alternatives when over capacity
- (-) Estimates can be inaccurate (KV cache varies)

### ADR-5: RLM for Complex Queries

**Context:** Users ask multi-step questions like "Can I run Qwen3-235B on 24GB?"
that require chaining model metadata, quantization options, driver flags, and
live VRAM data. No single MCP tool can answer these.

**Decision:** Use RLM (Recursive Language Models) with a custom `NonIsolatedEnv`
environment. The LLM explores data programmatically via a REPL, calling
whitelisted read-only MCP tools.

**Consequences:**

- (+) Programmatic exploration — LLM decides what to look up based on findings
- (+) Source citations — every recommendation traces to specific data
- (+) Read-only — returns advice, never executes mutations
- (-) Multiple LLM calls per query (cost, latency)
- (-) Requires LLM API access (not offline-capable)

**Reference:** See `research/rlm_langextract.md` in the repository.

### ADR-6: Function Whitelisting over Full Sandboxing

**Context:** RLM executes LLM-generated Python code. This is a security risk
requiring mitigation.

**Decision:** For v1, use function whitelisting (restricted namespace) plus AST
validation. Defer full sandbox-runtime (`srt`) isolation to a future release.

**Consequences:**

- (+) Simple to implement and reason about
- (+) No network/filesystem sandbox complications (fetchers need network)
- (+) Functions not in namespace raise `NameError` — clean failure mode
- (-) Less isolation than OS-level sandboxing
- (-) Relies on AST validation catching all dangerous patterns

**Future:** Evaluate sandbox-runtime (`srt`) with tool-based fetch pattern
for stronger isolation (tracked in gpumod-b9x).

---

## 10. Glossary

| Term            | Definition                                                   |
| --------------- | ------------------------------------------------------------ |
| **Mode**        | Named collection of services (e.g., "code", "rag", "blank")  |
| **Service**     | Managed ML process (e.g., vllm-chat, glm-code)               |
| **Driver**      | Runtime adapter implementing ServiceDriver ABC               |
| **Sleep Level** | VRAM optimization: L1 (fast wake), L2 (full release), Router |
| **Preset**      | YAML definition for a service configuration                  |
| **Simulation**  | Pre-flight VRAM calculation before deployment                |
| **VRAM**        | Video RAM on GPU (e.g., 24GB on RTX 4090)                    |
| **KV Cache**    | Key-Value cache for transformers (grows with context length) |
| **Orphan**      | Running service not in current mode's definition             |
| **MCP**         | Model Context Protocol for AI assistant integration          |
| **Discovery Layer** | Components that fetch external data (HF, GitHub) without LLM calls |
| **Consulting Layer** | RLM-based reasoning layer for complex multi-step queries    |
| **RLM**         | Recursive Language Model — treats context as a REPL variable |
| **NonIsolatedEnv** | RLM base class for custom environments with shared namespace |
| **Function Whitelisting** | Security pattern: only explicitly exposed functions are callable |
| **Knowledge Source Tier** | Trust/freshness classification (Core, Curated, Dynamic, Community) |

---

## References

- [arc42 Template](https://arc42.org/) — Architecture documentation structure
- [C4 Model](https://c4model.com/) — Software architecture visualization
- [MCP Specification](https://modelcontextprotocol.io/) — AI assistant integration

---

## See Also

- [Configuration Guide](../getting-started/configuration.md) — Environment variables and settings
- [CLI Reference](../getting-started/cli.md) — Command documentation
- [MCP Integration](../user-guide/mcp.md) — AI assistant setup
