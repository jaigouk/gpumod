# DockerDriver Design Document

> **Ticket:** gpumod-86k (P7-S0 SPIKE)
> **Status:** Complete
> **Author:** AI Architect Agent
> **Date:** 2026-02-07

---

## 1. Decision Record: Docker SDK vs Subprocess

### Decision

Use the **Docker SDK for Python** (`docker-py`) instead of subprocess calls to
`docker`/`podman`.

### Context

gpumod needs a `DockerDriver` to manage containerized services (e.g., Qdrant
vector database, Langfuse observability platform) alongside systemd-managed ML
inference services. The driver must implement the `ServiceDriver` ABC:
`start()`, `stop()`, `status()`, `health_check()`.

Two approaches were evaluated:

1. **Subprocess** (`asyncio.create_subprocess_exec` calling `docker` CLI)
2. **Docker SDK** (`docker-py` Python library)

### Evaluation

| Criterion | Subprocess | Docker SDK |
|-----------|-----------|------------|
| **Shell injection risk** | Requires input validation + command construction; one missed argument = injection vector | Eliminated entirely; typed Python API, no string commands |
| **Command surface area** | Docker CLI has 50+ subcommands with complex flag combinations; allowlist approach from `systemd.py` does not scale | SDK handles argument construction internally |
| **Type safety** | Returns raw strings; manual parsing required | Returns typed objects (`Container`, `Image`) with properties |
| **Health checks** | Must parse `docker inspect` JSON output | Native `.status`, `.attrs["State"]["Health"]` access |
| **Async support** | Naturally async via `create_subprocess_exec` | Sync library; requires `asyncio.to_thread()` wrapper |
| **Dependency weight** | No extra dependency | Adds `docker>=7.0,<8.0` (~2MB) |
| **Consistency** | Different pattern from other drivers (systemd uses subprocess but has tiny surface) | Aligns with "use the best tool" principle |
| **Podman compat** | Works with `podman` CLI directly | Works via `podman`'s Docker-compatible API socket |

### Rationale

The existing `systemd.py` uses subprocess safely because `systemctl` has only
8 allowlisted commands and a simple `unit_name` argument. Docker CLI has a
fundamentally larger attack surface (50+ commands, complex flags, volume specs,
environment variables, port mappings). Reproducing the systemd.py allowlist
pattern for Docker would require validating dozens of argument combinations --
brittle and error-prone.

The Docker SDK eliminates the entire class of shell/command injection attacks
by using a typed Python API. The tradeoff (sync library requiring
`asyncio.to_thread()`) is acceptable since container operations are infrequent
and already involve network I/O to the Docker daemon.

### Consequences

- Add `docker>=7.0,<8.0` to `pyproject.toml` dependencies with upper bound.
- Wrap all Docker SDK calls in `asyncio.to_thread()` for async compatibility.
- Mock `docker.DockerClient` in tests (no real Docker daemon needed for unit tests).

---

## 2. Security Threat Analysis

### 2.1 Container-Specific Threats

| # | Threat | Vector | Impact | Mitigation | Ref |
|---|--------|--------|--------|------------|-----|
| T18 | Image name injection | LLM passes `"image; rm -rf /"` as container image via MCP tool | Arbitrary command execution (if using subprocess); with SDK, API error but potential confusion | SEC-D7: Image name validated against strict regex before passing to SDK; reject names containing shell metacharacters | SEC-D7 |
| T19 | Volume mount path traversal | LLM passes `"../../etc/shadow:/data"` as volume mount in `extra_config` | Host filesystem read/write outside allowed directories | SEC-D8: Volume source paths resolved via `os.path.realpath()` and validated to be under explicitly allowed base directories (`$HOME`, `/tmp`); reject absolute paths outside allowlist | SEC-D8 |
| T20 | Privileged container escape | `extra_config` contains `privileged: true` or dangerous capabilities (`SYS_ADMIN`, `SYS_PTRACE`) | Full host access, container escape | SEC-D9: `privileged=False` hardcoded in driver; capability allowlist enforced (only `--gpus` via nvidia runtime); `pid_mode`, `network_mode="host"`, `ipc_mode="host"` blocked | SEC-D9 |
| T21 | Container name / env var injection | Malicious container name with ANSI escapes or env vars with sensitive data in `extra_config` | Terminal escape injection in CLI/TUI output; info disclosure if env vars leak | SEC-D10: Container names constructed from sanitized `service.id` with `gpumod-` prefix; env var keys validated against `^[A-Z_][A-Z0-9_]*$` regex; env var values sanitized (no newlines, no shell expansion) | SEC-D10 |

### 2.2 Additional Container Security Controls

**Docker Socket Access:**
- gpumod connects to the Docker daemon via the default socket (`/var/run/docker.sock`).
- The user running gpumod must be in the `docker` group or use rootless Docker.
- gpumod does NOT expose the Docker socket to managed containers (no DinD).

**GPU Access:**
- Containers requiring GPU access use `runtime="nvidia"` (nvidia-container-runtime).
- The `--gpus` flag is the only device-related option exposed.
- No `--device` passthrough for arbitrary host devices.

**Resource Limits:**
- Containers can have `mem_limit` set via `extra_config` to prevent host OOM.
- No CPU pinning unless explicitly configured.

**Image Pull Policy:**
- gpumod does NOT auto-pull images. Images must be pre-pulled on the host.
- This prevents supply-chain attacks via typosquatted image names.
- A future enhancement could add allowlisted registries.

---

## 3. Interface Design

### 3.1 DockerDriver Class

```python
class DockerDriver(ServiceDriver):
    """Driver for Docker-containerized services.

    Manages container lifecycle via the Docker SDK. Does not support
    sleep/wake -- containers use stop/start for memory release.

    Parameters
    ----------
    client:
        Optional Docker client for dependency injection (testing).
        Defaults to docker.from_env().
    http_timeout:
        Timeout in seconds for HTTP health check requests.
    container_prefix:
        Prefix for container names. Default "gpumod".
    """

    def __init__(
        self,
        client: docker.DockerClient | None = None,
        http_timeout: float = 10.0,
        container_prefix: str = "gpumod",
    ) -> None: ...

    async def start(self, service: Service) -> None: ...
    async def stop(self, service: Service) -> None: ...
    async def status(self, service: Service) -> ServiceStatus: ...
    async def health_check(self, service: Service) -> bool: ...

    @property
    def supports_sleep(self) -> bool:
        return False
```

### 3.2 Method Specifications

**`start(service)`**
1. Validate image name via `validate_docker_image()` (SEC-D7).
2. Validate volume mounts via `validate_volume_mounts()` (SEC-D8).
3. Construct container config: name=`{prefix}-{service.id}`, image from
   `extra_config["image"]`, ports/volumes/env from `extra_config`.
4. Hardcode `privileged=False`, apply capability restrictions (SEC-D9).
5. Call `client.containers.run(detach=True, ...)` via `asyncio.to_thread()`.

**`stop(service)`**
1. Look up container by name `{prefix}-{service.id}`.
2. Call `container.stop(timeout=30)` then `container.remove()`.
3. Handle `docker.errors.NotFound` gracefully (container already gone).

**`status(service)`**
1. Look up container by name.
2. Map Docker status to `ServiceState`:
   - `"running"` + healthy -> `ServiceState.RUNNING`
   - `"running"` + unhealthy -> `ServiceState.UNHEALTHY`
   - `"exited"` / `"created"` -> `ServiceState.STOPPED`
   - `"restarting"` -> `ServiceState.STARTING`
   - Not found -> `ServiceState.STOPPED`
   - Error -> `ServiceState.UNKNOWN`
3. Combine with HTTP health check (same pattern as VLLMDriver).

**`health_check(service)`**
1. Use `httpx.AsyncClient` to `GET http://localhost:{port}{health_endpoint}`.
2. Return `True` if status code 200, `False` otherwise.
3. Same implementation as VLLMDriver/FastAPIDriver (consistency).

### 3.3 Container Naming Convention

```
gpumod-{service.id}
```

- `service.id` is already validated by `validate_service_id()` (SEC-V1).
- The `gpumod-` prefix prevents name collisions with user containers.
- The full name is sanitized via `sanitize_name()` for display purposes.

### 3.4 Configuration via `extra_config`

The following keys are recognized in `service.extra_config` for Docker services:

| Key | Type | Required | Description | Validation |
|-----|------|----------|-------------|------------|
| `image` | `str` | Yes | Docker image name with optional tag | SEC-D7 regex |
| `volumes` | `dict[str, str]` | No | Host path -> container path mappings | SEC-D8 path validation |
| `environment` | `dict[str, str]` | No | Environment variables | SEC-D10 key/value validation |
| `ports` | `dict[str, int]` | No | Container port -> host port mappings | Port range 1-65535 |
| `command` | `str` | No | Override container CMD | No shell metacharacters |
| `runtime` | `str` | No | Container runtime (e.g., `"nvidia"`) | Allowlist: `"nvidia"`, `"runc"` |
| `mem_limit` | `str` | No | Memory limit (e.g., `"2g"`) | Regex: `^[0-9]+[kmgKMG]?$` |

These keys must be added to `EXTRA_CONFIG_ALLOWED_KEYS` in `validation.py`
(SEC-D6).

---

## 4. Validation Functions

Add to `src/gpumod/validation.py`:

```python
# Docker image name: lowercase alphanum, dots, hyphens, slashes, optional tag
DOCKER_IMAGE_RE = re.compile(
    r"^[a-z0-9][a-z0-9._/-]{0,127}"   # repository name
    r"(:[a-zA-Z0-9._-]{1,128})?$"      # optional tag
)

DOCKER_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")

ALLOWED_RUNTIMES: frozenset[str] = frozenset({"runc", "nvidia"})

VOLUME_ALLOWED_BASES: tuple[str, ...] = (
    os.path.expanduser("~"),
    "/tmp",
)


def validate_docker_image(value: str) -> str:
    """Validate a Docker image name (SEC-D7)."""
    ...

def validate_volume_mounts(volumes: dict[str, str]) -> dict[str, str]:
    """Validate volume mount paths are under allowed bases (SEC-D8)."""
    ...

def validate_docker_env(env: dict[str, str]) -> dict[str, str]:
    """Validate environment variable keys and values (SEC-D10)."""
    ...

def validate_container_runtime(runtime: str) -> str:
    """Validate container runtime against allowlist (SEC-D9)."""
    ...
```

---

## 5. Container Lifecycle

```
                    ┌─────────────┐
                    │   Service   │
                    │  (DB row)   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  start()    │
                    │             │
                    │ 1. Validate │
                    │ 2. Run      │
                    │ 3. Wait     │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │    Container Running    │
              │                        │
              │  health_check() polls  │
              │  status() inspects     │
              └────────────┬───────────┘
                           │
                    ┌──────▼──────┐
                    │   stop()    │
                    │             │
                    │ 1. Stop     │
                    │ 2. Remove   │
                    └─────────────┘
```

**Key behaviors:**
- `start()` is idempotent: if container already exists and is running, no-op.
- `stop()` is idempotent: if container does not exist, no-op.
- Containers are removed on stop (not just stopped) to avoid stale state.
- No restart policy is set by gpumod; container lifecycle is fully managed.

---

## 6. Test Strategy

### 6.1 Unit Tests (`tests/unit/test_docker_driver.py`)

Mock `docker.DockerClient` for all tests. Follow existing patterns from
`test_vllm_driver.py`:

| Test | Description | Security |
|------|-------------|----------|
| `test_start_runs_container` | Verify `client.containers.run()` called with correct args | |
| `test_start_validates_image` | Reject `"image; rm -rf /"` | SEC-D7 |
| `test_start_validates_volumes` | Reject `"../../etc/shadow:/data"` | SEC-D8 |
| `test_start_rejects_privileged` | Ensure `privileged=False` always set | SEC-D9 |
| `test_start_sanitizes_container_name` | Container name uses sanitized service.id | SEC-D10 |
| `test_start_validates_env_keys` | Reject env keys with special chars | SEC-D10 |
| `test_stop_removes_container` | Verify `container.stop()` + `container.remove()` | |
| `test_stop_handles_not_found` | No error when container already gone | |
| `test_status_maps_running` | Docker "running" -> ServiceState.RUNNING | |
| `test_status_maps_exited` | Docker "exited" -> ServiceState.STOPPED | |
| `test_status_maps_not_found` | NotFound -> ServiceState.STOPPED | |
| `test_status_maps_error` | Exception -> ServiceState.UNKNOWN | |
| `test_health_check_success` | HTTP 200 -> True | |
| `test_health_check_failure` | HTTP 500 / ConnectError -> False | |
| `test_supports_sleep_false` | `supports_sleep` returns `False` | |
| `test_rejects_host_network` | `network_mode="host"` blocked | SEC-D9 |
| `test_rejects_pid_host` | `pid_mode="host"` blocked | SEC-D9 |
| `test_validates_runtime_allowlist` | Only `"runc"` and `"nvidia"` accepted | SEC-D9 |
| `test_validates_port_range` | Ports must be 1-65535 | |
| `test_image_no_auto_pull` | Images must be pre-pulled | |

### 6.2 Integration Tests

- End-to-end container lifecycle with mocked Docker daemon.
- Security integration: injection attempts rejected at MCP boundary.

### 6.3 Mocking Pattern

```python
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_docker_client() -> MagicMock:
    client = MagicMock(spec=docker.DockerClient)
    client.containers = MagicMock()
    return client

@pytest.fixture
def driver(mock_docker_client: MagicMock) -> DockerDriver:
    return DockerDriver(client=mock_docker_client)
```

---

## 7. ServiceRegistry Integration

`DriverType.DOCKER` already exists in `src/gpumod/models.py`. The registry
needs one change in `src/gpumod/services/registry.py`:

```python
from gpumod.services.drivers.docker import DockerDriver

# In ServiceRegistry.__init__():
self._drivers: dict[DriverType, ServiceDriver] = {
    DriverType.VLLM: VLLMDriver(),
    DriverType.LLAMACPP: LlamaCppDriver(),
    DriverType.FASTAPI: FastAPIDriver(),
    DriverType.DOCKER: DockerDriver(),  # NEW
}
```

This follows the Open/Closed Principle: no changes to existing drivers.

---

## 8. Dependency Addition

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "docker>=7.0,<8.0",
]
```

The `docker` package is a well-maintained library (PyPI, Docker Inc.) with no
transitive security concerns beyond `requests` and `urllib3` which are
already common in Python ecosystems.

---

## 9. Out of Scope (for P7-T2 implementation)

- Docker Compose support (multi-container stacks).
- Image build from Dockerfile.
- Container log streaming to gpumod CLI (future enhancement).
- Registry authentication (image pull from private registries).
- Docker Swarm or Kubernetes integration.
- `podman` specific API differences (works via compatible socket).
