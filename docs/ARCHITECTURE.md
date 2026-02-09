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

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Assistant (Claude Code)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ stdio (JSON-RPC)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     gpumod MCP Server                           │
├─────────────────────────────────────────────────────────────────┤
│  Tools:                      Resources:                         │
│  - switch_mode               - gpumod://modes                   │
│  - gpu_status                - gpumod://services                │
│  - simulate_mode             - gpumod://models                  │
│  - start_service             - gpumod://help                    │
│  - stop_service                                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ServiceManager                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Cross-cutting Concepts

### VRAM Management

GPU VRAM is the central constraint. gpumod addresses this through:

1. **Pre-flight simulation** — Calculate VRAM before starting services
2. **Wait-for-release** — Poll nvidia-smi after stops until VRAM freed
3. **Sleep states** — L1/L2/Router reduce VRAM while keeping services warm
4. **Orphan detection** — Clean up services not in current mode definition

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

### Security

| Area            | Control                                                               |
| --------------- | --------------------------------------------------------------------- |
| **Docker**      | Privileged mode blocked, host network blocked, unsafe mounts rejected |
| **TUI**         | Untrusted text sanitized to prevent markup injection                  |
| **AI Planning** | LLM suggestions are advisory only, never auto-executed                |

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

---

## References

- [arc42 Template](https://arc42.org/) — Architecture documentation structure
- [C4 Model](https://c4model.com/) — Software architecture visualization
- [MCP Specification](https://modelcontextprotocol.io/) — AI assistant integration

---

## See Also

- [Configuration Guide](configuration.md) — Environment variables and settings
- [CLI Reference](cli.md) — Command documentation
- [Presets Guide](presets.md) — YAML service definitions
- [MCP Integration](mcp.md) — AI assistant setup
