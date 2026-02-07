---
title: Architecture Overview
description: System architecture for gpumod — service drivers, VRAM tracking, sleep management, mode switching, simulation engine, and MCP server design.
---

# gpumod Architecture

> **Version**: 0.1.0
> **Last Updated**: 2026-02-06

## Overview

**gpumod** is a GPU service management system designed for single-GPU Linux systems with systemd. It orchestrates multiple ML services (vLLM, llama.cpp, FastAPI) to maximize VRAM utilization through intelligent mode switching, sleep states, and pre-deployment simulation.

### Core Design Principles

1. **Services-First Architecture** - Everything revolves around service lifecycle management
2. **Database-Driven Configuration** - No hardcoded paths or service definitions
3. **VRAM-Aware Orchestration** - Simulate before deploy, prevent OOM
4. **Template-Based Deployment** - Jinja2 templates for systemd units and configs
5. **AI-Assisted Planning** - Interactive LLM chat for exploring model options

---

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│           CLI · Interactive TUI · MCP Server                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVICES LAYER (Core)                     │
├─────────────────────────────────────────────────────────────┤
│  ServiceManager: orchestrates all service operations        │
│  ├── ServiceRegistry: discovers & tracks services           │
│  ├── LifecycleManager: start/stop/restart with ordering     │
│  ├── HealthMonitor: continuous health checking              │
│  ├── VRAMTracker: real-time GPU memory tracking             │
│  └── SleepController: L1/L2/router sleep management         │
│                                                             │
│  Service Drivers (runtime-specific):                        │
│  ├── VLLMDriver: vLLM serve processes                       │
│  ├── LlamaCppDriver: llama.cpp server (router mode)         │
│  ├── FastAPIDriver: custom FastAPI servers (ASR, etc.)      │
│  └── DockerDriver: containerized services                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONFIGURATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  SQLite DB: services, modes, profiles, settings             │
│  Templates: Jinja2 for systemd units, configs               │
│  Presets: YAML definitions for common setups                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   SYSTEM LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  systemd: unit management (systemctl start/stop/status)     │
│  nvidia-smi: GPU info, VRAM usage                           │
│  HTTP: health endpoints, sleep/wake APIs                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Services Layer (Core)

### Service Abstraction

The service abstraction is the heart of gpumod. All GPU services (vLLM, llama.cpp, FastAPI) are represented uniformly.

**Key Types:**

```python
class ServiceState(Enum):
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SLEEPING = "sleeping"    # L1/L2 sleep
    UNHEALTHY = "unhealthy"  # Running but health check failed
    STOPPING = "stopping"
    FAILED = "failed"

@dataclass
class ServiceStatus:
    state: ServiceState
    vram_mb: int | None       # Current VRAM usage (from nvidia-smi)
    uptime_seconds: int | None
    health_ok: bool | None
    sleep_level: str | None   # "l1", "l2", "router", None
    last_error: str | None
```

### Service Drivers

Each service type has a specialized driver implementing the `ServiceDriver` ABC:

| Driver             | Sleep Support           | Health Check | Examples                  |
| ------------------ | ----------------------- | ------------ | ------------------------- |
| **VLLMDriver**     | L1, L2 (via API)        | `/health`    | vllm-embedding, vllm-chat |
| **LlamaCppDriver** | Router (load/unload)    | `/health`    | glm-code (devstral)       |
| **FastAPIDriver**  | Custom (if implemented) | Configurable | qwen3-asr, custom servers |
| **DockerDriver**   | Container stop/start    | Configurable | ollama, langfuse          |

**Driver Responsibilities:**

- Start/stop services via systemd or container runtime
- Check health via HTTP endpoints
- Manage sleep states (L1, L2, router)
- Report current VRAM usage
- Handle graceful shutdowns

### ServiceManager (Orchestrator)

The `ServiceManager` coordinates all service operations:

**Key Operations:**

- `switch_mode(target_mode)` - Switch between GPU modes with pre-flight VRAM checks
- `get_status()` - Full system status (GPU, VRAM, services)
- `start_service(service_id)` - Start with dependency ordering
- `stop_service(service_id)` - Stop with dependent handling

**Pre-Flight Checks:**
Before switching modes, ServiceManager:

1. Calculates projected VRAM usage
2. Compares against GPU capacity
3. Returns alternatives if exceeds
4. Only proceeds if safe

### Lifecycle Management

The `LifecycleManager` handles service startup/shutdown with dependency ordering:

**Dependency Resolution:**

- Services can declare `depends_on` (e.g., Chat depends on Embedding)
- Topological sort ensures correct startup order
- Reverse order for shutdown (dependents stop first)

**Health Waiting:**

- After starting, waits for health endpoint to respond
- Configurable timeout per service
- Fails fast on unhealthy starts

### VRAM Tracking

The `VRAMTracker` monitors GPU memory usage:

**Data Sources:**

- `nvidia-smi --query-gpu=memory.used,memory.free` for totals
- `nvidia-smi -q -x` for per-process breakdown
- Service configs for estimated VRAM (when idle)

**Estimation:**
For services not yet running, VRAM is estimated from:

1. Stored `vram_mb` in service config
2. Model info (weights + KV cache calculation)
3. Historical usage patterns

### Sleep Management

The `SleepController` manages GPU memory optimization:

**Sleep Levels:**
| Level | Wake Time | VRAM Saved | Use Case |
| ----- | --------- | ---------- | -------- |
| **L1** | ~9ms | Partial (keeps CUDA context) | Frequent wake (Chat, ASR) |
| **L2** | ~1-2s | Full (discards everything) | Rare wake (HyDE, Reranker) |
| **Router** | ~500ms | Full (unload model) | llama.cpp model swapping |

**Auto-Sleep:**
Services can be configured to sleep after idle timeout:

```python
await sleep_controller.auto_sleep_idle(idle_timeout=300)
```

### Health Monitoring

The `HealthMonitor` provides continuous health checking for running services:

**Interface:**
- `start_monitoring(service_id)` -- Begin monitoring a service
- `stop_monitoring(service_id)` -- Stop monitoring a service
- `get_health(service_id)` -- Get current health status

**Design:**
- One `asyncio.Task` per monitored service
- Configurable check interval and failure threshold
- Exponential backoff with jitter on repeated failures
- Debounce: state transitions require consecutive failures/successes
- Automatic cleanup of monitoring tasks on shutdown

**Health States:**
| State | Meaning |
| ----- | ------- |
| **healthy** | Service responding to health endpoint |
| **unhealthy** | Consecutive failures exceeded threshold |
| **unknown** | Not yet checked or monitoring stopped |

### DockerDriver

The `DockerDriver` manages containerized services via the Docker SDK
(`docker-py`). Unlike systemd-based drivers, it creates and manages
containers directly.

**Container Lifecycle:**
1. `start` -- Pull image (if needed), create container, start
2. `stop` -- Stop container with graceful timeout, remove
3. `status` -- Inspect container state via Docker API
4. `health` -- HTTP health check to `localhost:{port}{health_endpoint}`

**Security Controls:**
| Control | Description |
| ------- | ----------- |
| **SEC-D7** | `--privileged` mode blocked |
| **SEC-D8** | Host/macvlan network modes blocked |
| **SEC-D9** | Unsafe volume mounts rejected (`/`, `/etc`, `/var/run/docker.sock`) |
| **SEC-D10** | Environment variables sanitized (no `=` in keys) |

**Configuration:** Docker-specific settings are stored in
`extra_config` on the `Service` model: `image`, `ports`, `environment`,
`volumes`, `command`.

### Interactive TUI

The `GpumodApp` is a Textual-based terminal dashboard providing a live
view of GPU status and service state.

**Widgets:**
- `GPUBar` -- ASCII VRAM usage bar with percentage
- `ServiceList` -- Service table with state indicators
- `OutputPanel` -- Scrollable command output area
- `HelpBar` -- Footer with keyboard shortcuts

**Commands:** `/status`, `/switch <mode>`, `/simulate`, `/quit`

**Security:** All untrusted text (service names, model IDs) is rendered
via `rich.text.Text` with `sanitize_name()` to prevent markup injection.

---

## Configuration Layer

### Database Schema

**Core Tables:**

```sql
-- GPU hardware profiles
CREATE TABLE gpu_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    vram_mb INTEGER NOT NULL,
    architecture TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Service definitions
CREATE TABLE services (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- "vllm", "llamacpp", "fastapi", "docker"
    port INTEGER,
    vram_mb INTEGER NOT NULL,
    sleep_mode TEXT,  -- "none", "l1", "l2", "router"
    health_endpoint TEXT DEFAULT "/health",
    model_id TEXT,
    extra_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mode definitions (groups of services)
CREATE TABLE modes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    total_vram_mb INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many: mode ↔ services
CREATE TABLE mode_services (
    mode_id TEXT REFERENCES modes(id),
    service_id TEXT REFERENCES services(id),
    start_order INTEGER DEFAULT 0,
    sleep_on_idle BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (mode_id, service_id)
);

-- Systemd unit templates
CREATE TABLE service_templates (
    service_id TEXT PRIMARY KEY REFERENCES services(id),
    unit_template TEXT NOT NULL,
    preset_template TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User settings
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model registry (for VRAM estimation)
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,  -- "huggingface", "gguf", "local"
    parameters_b REAL,
    architecture TEXT,
    base_vram_mb INTEGER,
    kv_cache_per_1k_tokens_mb INTEGER,
    quantizations JSON,
    capabilities JSON,
    fetched_at TIMESTAMP,
    notes TEXT
);

-- Simulation history
CREATE TABLE simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode_id TEXT,
    services JSON,
    total_vram_mb INTEGER,
    fits BOOLEAN,
    alternatives JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Template System

**Systemd Unit Templates (Jinja2):**

```jinja2
[Unit]
Description={{ service.name }}
After=network.target

[Service]
Type=simple
User={{ settings.user }}
Environment="CUDA_VISIBLE_DEVICES={{ settings.cuda_devices | default('0') }}"
Environment="HF_HOME={{ settings.hf_home }}"

ExecStart={{ settings.vllm_bin }} serve \
    {{ service.model_id }} \
    --port {{ service.port }} \
    --host 0.0.0.0 \
    --gpu-memory-utilization {{ service.gpu_mem_util }} \
    --max-model-len {{ service.max_model_len }}

Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**Service Presets (YAML):**

```yaml
# presets/llm/devstral-small-2.yaml
id: glm-code
name: "Devstral Small 2 (24B)"
driver: llamacpp
port: 7070

vram_mb: 19100
context_size: 32768
kv_cache_per_1k: 170

model_id: "mistralai/Devstral-Small-2-24B-Instruct"
model_path: "$MODELS_DIR/Devstral-Small-2-24B-Q4_K_M.gguf"

health_endpoint: /health
startup_timeout: 60
supports_sleep: true
sleep_mode: router

unit_template: llamacpp.service.j2
unit_vars:
  preset_file: llama_cpp_gguf_presets.ini
  flash_attn: true
```

---

## Simulation & Visualization

### Pre-Simulation Engine

Before deploying services, gpumod simulates VRAM requirements:

**Simulation Flow:**

1. Fetch current VRAM usage
2. Calculate target mode requirements
3. Compare against GPU capacity
4. Generate alternatives if exceeds

**ASCII Visualization:**

```
                    RTX 4090 VRAM (24GB)
    0GB        6GB        12GB        18GB        24GB
    |----------|----------|----------|----------|

current (code mode):
    [██████████████████████████████████████████░░] 21.7GB
    |--EmbCode---|--------Devstral (32K)-----------|
       2.5GB              +19.1GB

proposed (add reranker):
    [██████████████████████████████████████████████████] 27.7GB ❌
    |--EmbCode---|--------Devstral (32K)-----------|--Rerank--|
       2.5GB              +19.1GB                    +6GB

    ⚠️  Exceeds 24GB by 3.7GB. Alternatives:
    - Switch to Devstral 16K (-3GB KV cache)
    - Use reranker with L2 sleep (time-share)
    - Offload reranker to CPU (slow but fits)
```

### Model Info Fetching

**Data Sources:**

| Source          | API               | Data Retrieved                                     |
| --------------- | ----------------- | -------------------------------------------------- |
| **HuggingFace** | `huggingface_hub` | Model config, parameter count, safetensors size    |
| **Ollama**      | `GET /api/show`   | GGUF quantization, actual VRAM from running models |

**VRAM Estimation:**

```python
base_vram_mb = model_size_mb  # Weights
kv_cache_mb = (context_size / 1000) * kv_cache_per_1k
total_vram = base_vram_mb + kv_cache_mb
```

---

## Mode Switching

### Mode Definition

A **mode** is a named collection of services for a specific use case:

| Mode      | Services                                        | VRAM         | Use Case            |
| --------- | ----------------------------------------------- | ------------ | ------------------- |
| **code**  | embedding-code, glm-code                        | ~22GB        | Agentic coding      |
| **rag**   | embedding-code, embedding, hyde, reranker, chat | ~13.5GB peak | RAG pipeline        |
| **speak** | embedding, asr, tts, chat                       | ~23GB        | Voice conversations |
| **blank** | (none)                                          | 0GB          | Manual GPU usage    |

### Switch Workflow

```
1. User requests: gpumod mode switch rag
2. ServiceManager validates:
   - Mode exists in DB
   - VRAM requirements fit
3. Calculate service diff:
   - to_stop = current - target
   - to_start = target - current
4. Pre-flight check:
   - Simulate VRAM with target services
   - Return error if exceeds
5. Execute switch:
   - Stop unneeded services (reverse dependency order)
   - Start new services (dependency order)
   - Wait for health checks
6. Update current mode in DB
7. Return result with services started/stopped
```

### Time-Shared Modes

For modes exceeding VRAM, services can time-share via sleep:

**Example: RAG Pipeline**

```
Idle: 6.5GB (embedding-code + embedding-vl always awake)

Query arrives:
1. Wake HyDE → expand query → sleep HyDE
   Peak: 11.5GB
2. Embed query (always awake)
   Peak: 6.5GB
3. Wake Reranker → rerank results → sleep Reranker
   Peak: 12.5GB
4. Wake Chat → generate response → sleep Chat
   Peak: 13.5GB

Max peak: 13.5GB (well within 24GB)
```

---

## CLI & Interactive Mode

### CLI Commands

```bash
# Mode management
gpumod mode list                     # List all modes
gpumod mode switch <mode>            # Switch GPU mode
gpumod mode status                   # Current mode status
gpumod mode create <name> --services a,b,c

# Service management
gpumod service list                  # All services
gpumod service status <id>           # Specific service
gpumod service start <id>            # Start service
gpumod service stop <id>             # Stop service

# Simulation
gpumod simulate --mode <mode>        # Simulate mode
gpumod simulate --add <service>      # Add to current mode
gpumod simulate --services a,b,c     # Custom combination

# Model info
gpumod model info <model-id>         # Fetch from HF/Ollama
gpumod model search <query>          # Search HuggingFace

# AI planning
gpumod plan                          # Interactive AI chat
gpumod plan "add vision model"       # One-shot query

# System
gpumod status                        # Full system status
gpumod status --visual               # ASCII dashboard
gpumod init                          # First-time setup
```

### Interactive Mode (Textual)

```
┌─────────────────────────────── gpumod ────────────────────────────────┐
│ ▐█▌ RTX 4090 (24GB) │ code │ VRAM: 21.7/24GB [██████████░░] 90%       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ● vllm-embedding-code   2.5GB  running                               │
│  ● glm-code (devstral)  19.1GB  running  32K ctx                      │
│  ○ vllm-embedding        stopped                                      │
│  ○ qwen3-asr             stopped                                      │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│ > /simulate --add qwen3-reranker                                      │
│                                                                       │
│ 🔍 Simulating: code + qwen3-reranker                                  │
│   Current:   21.6GB ✓                                                 │
│   Proposed:  27.6GB ❌ exceeds by 3.6GB                               │
│                                                                       │
│   Alternatives:                                                       │
│   [1] Use L2 sleep (time-share)                                       │
│   [2] Reduce context: 32K→16K (-3GB)                                  │
│   [3] Switch to bge-reranker-base (1.5GB)                             │
│                                                                       │
│ > _                                                                   │
├───────────────────────────────────────────────────────────────────────┤
│ ?: help │ /status │ /switch <mode> │ /simulate │ q: quit              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## MCP Server Integration

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
├───────────────┬───────────────┬─────────────────────────────┤
│  CLI Commands │ Interactive   │  External MCP Clients       │
│  gpumod ...   │ /slash cmds   │  (Claude Code, etc.)        │
└───────┬───────┴───────┬───────┴──────────────┬──────────────┘
        │               │                      │
        └───────────────┴──────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │         MCP Server (FastMCP)      │
        ├───────────────────────────────────┤
        │  Tools:                           │
        │  - gpu_mode_switch(mode)          │
        │  - gpu_mode_status()              │
        │  - simulate_mode(mode, add=[])    │
        │  - model_info(model_id)           │
        │  - service_list()                 │
        │  - plan_with_ai(prompt)           │
        └───────────────────────────────────┘
```

**Key Principle:** All user-facing operations (CLI, interactive, external MCP) call the same MCP tools internally. This ensures consistency and a single source of truth.

### MCP Tools

| Tool              | Purpose               | Example                                       |
| ----------------- | --------------------- | --------------------------------------------- |
| `gpu_mode_switch` | Switch to a mode      | `{"mode": "rag"}`                             |
| `gpu_mode_status` | Get current status    | `{}`                                          |
| `gpu_mode_verify` | Verify mode health    | `{"mode": "rag"}`                             |
| `simulate_mode`   | Pre-flight simulation | `{"mode": "code", "add": ["qwen3-reranker"]}` |
| `model_info`      | Fetch model metadata  | `{"model_id": "Qwen/Qwen3-VL-2B"}`            |
| `service_list`    | List all services     | `{}`                                          |
| `service_status`  | Get service status    | `{"service_id": "vllm-chat"}`                 |
| `plan_with_ai`    | AI planning chat      | `{"prompt": "add vision model"}`              |

---

## Platform Requirements

### Supported

- **OS**: Linux (tested on Ubuntu 22.04, 24.04)
- **Init**: systemd (required for service management)
- **GPU**: NVIDIA with CUDA support (nvidia-smi required)
- **Python**: 3.11+

### Out of Scope (v0.1)

| Feature        | Why                          | Future?                     |
| -------------- | ---------------------------- | --------------------------- |
| macOS/Windows  | No systemd                   | Maybe v0.3 with launchd/WSL |
| AMD/Intel GPUs | Different tooling (rocm-smi) | Maybe v0.2                  |
| Multi-GPU      | Complexity, rare use case    | v0.2                        |
| Remote GPU     | Network latency, auth        | v0.2                        |

---

## Directory Structure

```
gpumod/
├── src/
│   └── gpumod/
│       ├── __init__.py
│       ├── cli.py              # Click-based CLI
│       ├── db.py               # SQLite operations
│       ├── models.py           # Pydantic models
│       ├── tui.py              # Interactive Textual TUI
│       ├── services/           # Service management
│       │   ├── base.py         # ServiceDriver ABC
│       │   ├── manager.py      # ServiceManager orchestrator
│       │   ├── registry.py     # ServiceRegistry
│       │   ├── lifecycle.py    # LifecycleManager
│       │   ├── vram.py         # VRAMTracker
│       │   ├── sleep.py        # SleepController
│       │   ├── health.py       # HealthMonitor
│       │   └── drivers/        # Service drivers
│       │       ├── vllm.py     # VLLMDriver
│       │       ├── llamacpp.py # LlamaCppDriver
│       │       ├── fastapi.py  # FastAPIDriver
│       │       └── docker.py   # DockerDriver
│       ├── templates/          # Jinja2 templates
│       │   ├── systemd/
│       │   │   ├── vllm.service.j2
│       │   │   ├── llamacpp.service.j2
│       │   │   └── fastapi.service.j2
│       │   └── config/
│       │       ├── lmcache.yaml.j2
│       │       └── llamacpp-preset.ini.j2
│       ├── server.py           # MCP server
│       ├── simulation.py       # Pre-simulation engine
│       ├── visualization.py    # ASCII VRAM visualization
│       └── planning.py         # AI-assisted planning
├── migrations/                 # Database migrations
├── presets/                    # Built-in service presets
│   ├── embedding/
│   ├── llm/
│   ├── audio/
│   └── modes/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── ARCHITECTURE.md         # This file
│   ├── configuration.md
│   ├── templates.md
│   └── presets.md
├── pyproject.toml
└── README.md
```

---

## Testing Strategy

### Unit Tests (GPU-free)

```python
# tests/unit/test_simulation.py
def test_simulate_exceeds_vram(mock_db):
    """Simulation correctly detects VRAM overflow."""
    mock_db.add_service("svc1", vram_mb=15000)
    mock_db.add_service("svc2", vram_mb=15000)
    mock_db.set_profile(vram_mb=24000)

    result = simulate_services(["svc1", "svc2"])

    assert not result.fits
    assert result.exceeds_by_mb == 6000
    assert len(result.alternatives) > 0
```

### Integration Tests (Mocked systemd)

```python
# tests/integration/test_mode_switch.py
@pytest.fixture
def mock_systemctl(monkeypatch):
    """Mock systemctl for testing mode switches."""
    calls = []
    async def fake_run(cmd):
        calls.append(cmd)
        return CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("gpumod.services.systemd.run", fake_run)
    return calls

async def test_mode_switch_stops_old_services(mock_systemctl, mock_db):
    mock_db.set_current_mode("code")
    await switch_mode("rag")

    assert "stop glm-code.service" in " ".join(mock_systemctl)
    assert "start vllm-embedding.service" in " ".join(mock_systemctl)
```

### E2E Tests (Real GPU/Docker, CI-optional)

E2E tests use real SQLite databases and exercise the full service layer
stack. GPU and Docker-dependent tests are marked with custom pytest
markers and skip gracefully on machines without the required hardware.

```python
# tests/e2e/conftest.py detects GPU and Docker availability:
# - @pytest.mark.gpu_required  -- skipped when nvidia-smi unavailable
# - @pytest.mark.docker_required -- skipped when Docker daemon unavailable

# CI configurations:
# CPU-only CI:   pytest tests/ -m "not gpu_required and not docker_required"
# GPU CI:        pytest tests/  (runs everything)
# Docker CI:     pytest tests/ -m "not gpu_required"
```

**E2E coverage:**
- Mode switch lifecycle (DB + registry + lifecycle manager)
- VRAM simulation with real DB and mocked GPU info
- Service status through the full stack
- GPU detection via `nvidia-smi` (on GPU machines)
- Docker container status via Docker SDK (on Docker machines)
- Fixture cleanup verification

---

## Design Decisions

### Why SQLite?

- **No daemon**: Embedded, zero-setup
- **Transactional**: ACID guarantees for mode switches
- **Portable**: Single file, easy backup
- **Fast**: Sub-millisecond queries for our scale

### Why Jinja2 Templates?

- **Flexibility**: Users can customize without Python
- **Readability**: Systemd units are self-documenting
- **Reusable**: Same template for multiple services
- **Version Control**: Templates are text files

### Why Service Drivers?

- **Abstraction**: Hide vLLM/llama.cpp/FastAPI differences
- **Testability**: Mock drivers for unit tests
- **Extensibility**: Add new runtimes without core changes
- **Sleep Management**: Unified interface for L1/L2/router

### Why Pre-Simulation?

**Problem:** Deploying a new service to find it OOMs wastes time and disrupts running services.

**Solution:** Simulate VRAM requirements before `systemctl start`. Calculate total usage, compare against GPU capacity, suggest alternatives if exceeds.

**Impact:** Zero-downtime experimentation. Try before you buy.

---

## Future Enhancements

### v0.2 (Multi-GPU)

- Support for multiple GPUs via `CUDA_VISIBLE_DEVICES`
- Spread services across GPUs
- Cross-GPU dependency management

### v0.3 (Cross-Platform)

- macOS support via launchd
- Windows support via WSL2 + systemd
- AMD GPU support via rocm-smi

### v0.4 (Kubernetes)

- Separate project: gpumod-k8s
- Helm charts for GPU node pools
- Horizontal scaling for inference

---

## Contributing

See [Contributing](contributing.md) for development setup, testing guidelines, and PR process.

---

## License

Apache License 2.0

---

## Glossary

- **Mode**: Named collection of services (e.g., "code", "rag")
- **Service**: Managed process (e.g., vllm-chat, glm-code)
- **Driver**: Runtime adapter (VLLMDriver, LlamaCppDriver)
- **Sleep Level**: VRAM optimization technique (L1, L2, router)
- **Preset**: YAML template for common service configs
- **Simulation**: Pre-flight VRAM calculation before deploy
- **VRAM**: Video RAM on GPU (24GB on RTX 4090)
- **KV Cache**: Key-Value cache for transformer models (grows with context size)
- **Time-Sharing**: Sequential service wake/sleep to exceed GPU VRAM
