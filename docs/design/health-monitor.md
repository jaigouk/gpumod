# HealthMonitor Design Document

> **Ticket:** gpumod-8pg (P7-S1 SPIKE)
> **Status:** Complete
> **Author:** AI Architect Agent
> **Date:** 2026-02-07

---

## 1. Decision Record: Polling Model

### Decision

Use **periodic asyncio.Task polling** with per-service intervals, jitter, and
consecutive-failure thresholds.

### Context

gpumod's `LifecycleManager._wait_for_healthy()` (lifecycle.py:113) currently
does one-shot polling: it loops until a service becomes healthy during startup,
then stops. There is no continuous monitoring after startup.

The architecture (ARCHITECTURE.md:35) requires a `HealthMonitor` component for
**continuous** health checking — detecting services that become unhealthy after
initial startup and reporting state changes to the ServiceManager.

Three approaches were evaluated:

1. **Single-task polling loop** — one asyncio.Task iterates all services sequentially.
2. **Per-service asyncio.Task** — one asyncio.Task per monitored service.
3. **Event-driven** — services push health status via webhooks/events.

### Evaluation

| Criterion | Single-task loop | Per-service tasks | Event-driven |
|-----------|-----------------|-------------------|--------------|
| **Isolation** | One slow health check blocks all others | Independent; slow service doesn't block others | Full isolation |
| **Complexity** | Simple | Moderate (task lifecycle management) | High (requires service cooperation) |
| **Failure detection latency** | Proportional to number of services × interval | Constant per service | Near-instant |
| **Driver compatibility** | Works with existing `health_check()` ABC | Works with existing `health_check()` ABC | Requires new protocol in every driver |
| **Resource usage** | 1 task, sequential I/O | N tasks (N = services), concurrent I/O | 0 polling tasks, but webhook server needed |
| **Backoff support** | Global only | Per-service | N/A |

### Rationale

**Per-service asyncio.Task** is the best fit because:
- Health checks are I/O-bound (HTTP requests); concurrent execution avoids head-of-line blocking.
- Different services may need different intervals (e.g., a slow-starting model vs a lightweight sidecar).
- Per-service backoff on failure is straightforward — each task manages its own state.
- No changes to the existing `ServiceDriver` ABC — reuses `health_check()`.
- Bounded resource usage: gpumod manages ~5-10 services, so ~5-10 lightweight asyncio.Tasks.

Event-driven was rejected because it would require adding health-push protocol to
every driver (VLLM, LlamaCpp, FastAPI, Docker) — violating Open/Closed Principle and
significantly increasing scope.

### Consequences

- HealthMonitor owns task creation/cancellation for each monitored service.
- LifecycleManager remains responsible for startup health waiting (no change).
- ServiceManager wires HealthMonitor and reacts to health state changes.

---

## 2. Security Threat Analysis

### 2.1 Health Monitoring Threats

| # | Threat | Vector | Impact | Mitigation | Ref |
|---|--------|--------|--------|------------|-----|
| T22 | Health endpoint SSRF | Service `port` or `health_endpoint` configured to point at internal host (e.g., `http://169.254.169.254/metadata`) | Internal network scanning, cloud metadata exfiltration | SEC-H1: Health checks only connect to `localhost` (hardcoded in all existing drivers); `health_endpoint` validated by SEC-V1 regex (no `://` or hostname allowed) | SEC-H1 |
| T23 | Response parsing injection | Health endpoint returns crafted JSON/HTML that triggers parser vulnerability | Code execution or memory corruption in httpx/pydantic | SEC-H2: Health check only inspects HTTP status code (200 = healthy); response body is never parsed, deserialized, or logged | SEC-H2 |
| T24 | DoS via rapid health checks | HealthMonitor polls too aggressively, overwhelming the service | Service degradation, increased latency for real requests | SEC-H3: Minimum poll interval enforced (floor of 5 seconds); jitter prevents thundering herd; backoff on failures reduces load on struggling services | SEC-H3 |
| T25 | Resource exhaustion via stuck health tasks | Health check hangs (connection never closes), accumulating blocked tasks | asyncio event loop starvation, memory leak | SEC-H4: Per-request timeout via `httpx.AsyncClient(timeout=...)` (existing pattern); asyncio.wait_for wraps the entire check cycle | SEC-H4 |
| T26 | Health state manipulation | Attacker controls a service and alternates healthy/unhealthy responses to trigger rapid state oscillations | Log spam, alert fatigue, potential cascading restarts | SEC-H5: Consecutive-failure threshold (service must fail N checks before declared unhealthy); consecutive-success threshold for recovery; debounce prevents rapid state transitions | SEC-H5 |

### 2.2 Security Controls Summary

| Control | Description | Implementation |
|---------|-------------|----------------|
| SEC-H1 | Localhost-only health checks | Hardcoded `http://localhost:{port}` in all drivers; health_endpoint validated at DB boundary |
| SEC-H2 | Status-code-only inspection | `health_check()` returns `bool` from status code; no body parsing |
| SEC-H3 | Minimum poll interval | `HealthMonitor` enforces `min_interval=5.0` seconds; constructor validates |
| SEC-H4 | Per-check timeout | `asyncio.wait_for(driver.health_check(...), timeout=check_timeout)` |
| SEC-H5 | Consecutive-failure debounce | `failure_threshold=3` before state change; `recovery_threshold=2` before re-healthy |

---

## 3. Interface Design

### 3.1 HealthMonitor Class

```python
class HealthMonitor:
    """Continuous health monitoring for registered services.

    Single Responsibility: monitors health and reports state changes.
    Does NOT manage lifecycle (start/stop) — that's LifecycleManager's job.

    Parameters
    ----------
    registry:
        ServiceRegistry for looking up services and drivers.
    on_state_change:
        Callback invoked when a service's health state changes.
        Signature: (service_id: str, healthy: bool) -> None
    default_interval:
        Default polling interval in seconds.
    failure_threshold:
        Number of consecutive failures before declaring unhealthy.
    recovery_threshold:
        Number of consecutive successes before declaring healthy again.
    check_timeout:
        Per-health-check timeout in seconds.
    min_interval:
        Minimum allowed polling interval (SEC-H3).
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        on_state_change: Callable[[str, bool], Awaitable[None]] | None = None,
        default_interval: float = 15.0,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        check_timeout: float = 10.0,
        min_interval: float = 5.0,
    ) -> None: ...

    async def start_monitoring(self, service_id: str, interval: float | None = None) -> None:
        """Begin health monitoring for a service. Idempotent."""
        ...

    async def stop_monitoring(self, service_id: str) -> None:
        """Stop health monitoring for a service. Idempotent."""
        ...

    async def stop_all(self) -> None:
        """Stop monitoring all services and cancel all tasks."""
        ...

    def get_health_status(self, service_id: str) -> ServiceHealthInfo | None:
        """Get the current health status of a monitored service."""
        ...

    @property
    def monitored_services(self) -> frozenset[str]:
        """Set of service IDs currently being monitored."""
        ...
```

### 3.2 ServiceHealthInfo Dataclass

```python
@dataclass(frozen=True)
class ServiceHealthInfo:
    """Snapshot of a service's health monitoring state."""

    service_id: str
    healthy: bool
    consecutive_failures: int
    consecutive_successes: int
    last_check_at: float  # time.monotonic()
    last_healthy_at: float | None
    last_unhealthy_at: float | None
```

### 3.3 Internal: _ServiceHealthTask

```python
class _ServiceHealthTask:
    """Internal state for a single service's health monitoring task.

    Not part of the public API.
    """

    service_id: str
    interval: float
    failure_threshold: int
    recovery_threshold: int
    check_timeout: float

    # Mutable state
    task: asyncio.Task[None] | None
    consecutive_failures: int
    consecutive_successes: int
    healthy: bool
    last_check_at: float
    last_healthy_at: float | None
    last_unhealthy_at: float | None
```

---

## 4. Failure Detection Strategy

### 4.1 Consecutive-Failure Threshold

A single failed health check does NOT mark a service as unhealthy. Instead:

```
healthy → unhealthy: requires `failure_threshold` consecutive failures (default 3)
unhealthy → healthy: requires `recovery_threshold` consecutive successes (default 2)
```

This prevents flapping on transient network blips or temporary load spikes.

### 4.2 Jitter

Each poll interval has random jitter of ±20% to prevent thundering herd:

```python
actual_interval = interval * (0.8 + random.random() * 0.4)
```

This ensures that if 10 services all have `interval=15s`, their health checks
are spread across a ~6-second window rather than hitting simultaneously.

### 4.3 Exponential Backoff on Failure

When a service is unhealthy, the poll interval increases exponentially to
reduce load on a struggling service:

```python
backoff_interval = min(interval * (2 ** consecutive_failures), max_backoff)
```

With `max_backoff = 120.0` seconds. Once the service recovers (passes
`recovery_threshold` checks), the interval resets to the configured value.

### 4.4 State Transition Diagram

```
                    start_monitoring()
                          │
                          ▼
                   ┌─────────────┐
                   │   HEALTHY   │◄──────────────────┐
                   │             │                    │
                   └──────┬──────┘                    │
                          │                           │
                    health_check()                    │
                    returns False                     │
                          │                      consecutive_successes
                          ▼                      >= recovery_threshold
                   ┌─────────────┐                    │
                   │  DEGRADED   │                    │
                   │ (1-2 fails) │                    │
                   └──────┬──────┘                    │
                          │                           │
                    consecutive_failures              │
                    >= failure_threshold               │
                          │                           │
                          ▼                           │
                   ┌─────────────┐                    │
                   │  UNHEALTHY  │────────────────────┘
                   │ (backoff)   │  health_check() returns True
                   └─────────────┘
```

### 4.5 Timeout Handling

Each health check is wrapped in `asyncio.wait_for()`:

```python
try:
    healthy = await asyncio.wait_for(
        driver.health_check(service),
        timeout=self._check_timeout,
    )
except asyncio.TimeoutError:
    healthy = False  # treat timeout as failure
```

This prevents hung connections from blocking the monitoring task (SEC-H4).

---

## 5. Integration with ServiceManager

### 5.1 Wiring

```python
class ServiceManager:
    def __init__(
        self,
        db: Database,
        registry: ServiceRegistry,
        lifecycle: LifecycleManager,
        vram: VRAMTracker,
        sleep: SleepController,
        health: HealthMonitor | None = None,  # NEW (optional for backward compat)
    ) -> None:
        self._health = health or HealthMonitor(
            registry=registry,
            on_state_change=self._on_health_change,
        )

    async def _on_health_change(self, service_id: str, healthy: bool) -> None:
        """React to health state changes reported by HealthMonitor."""
        if healthy:
            logger.info("Service %r recovered", service_id)
        else:
            logger.warning("Service %r is unhealthy", service_id)
        # Future: auto-restart, alerting, mode degradation
```

### 5.2 Lifecycle Integration

- `ServiceManager.start_service()` → after `LifecycleManager.start()` completes → `HealthMonitor.start_monitoring()`
- `ServiceManager.stop_service()` → `HealthMonitor.stop_monitoring()` → then `LifecycleManager.stop()`
- `ServiceManager.shutdown()` → `HealthMonitor.stop_all()`

### 5.3 Backward Compatibility

The `health` parameter defaults to `None` for backward compatibility.
Existing tests and code that don't pass a HealthMonitor will get a default
instance created internally. The `on_state_change` callback is optional.

---

## 6. SOLID Analysis

| Principle | How HealthMonitor follows it |
|-----------|------|
| **Single Responsibility** | HealthMonitor ONLY monitors health and reports changes. It does not start/stop services (LifecycleManager), track VRAM (VRAMTracker), or manage sleep (SleepController). |
| **Open/Closed** | Adding a new driver (e.g., DockerDriver) requires zero changes to HealthMonitor — it uses the `ServiceDriver.health_check()` ABC method. |
| **Liskov Substitution** | All ServiceDriver implementations provide `health_check()` with the same contract (returns bool). HealthMonitor is agnostic to the concrete driver. |
| **Interface Segregation** | HealthMonitor depends only on `ServiceRegistry` (for lookups) and `ServiceDriver.health_check()` (for probing). No fat interfaces. |
| **Dependency Inversion** | HealthMonitor depends on the `ServiceDriver` abstraction, not concrete drivers. The `on_state_change` callback decouples it from ServiceManager's reaction logic. |

---

## 7. Test Strategy

### 7.1 Unit Tests (`tests/unit/test_health_monitor.py`)

| Test | Description | Security |
|------|-------------|----------|
| `test_start_monitoring_creates_task` | Task created for service | |
| `test_start_monitoring_idempotent` | Second call is no-op | |
| `test_stop_monitoring_cancels_task` | Task cancelled and removed | |
| `test_stop_monitoring_idempotent` | No error when not monitoring | |
| `test_stop_all_cancels_all_tasks` | All tasks cancelled | |
| `test_healthy_service_stays_healthy` | Consistent True → no state change callback | |
| `test_single_failure_no_state_change` | One failure doesn't trigger callback (threshold) | SEC-H5 |
| `test_consecutive_failures_trigger_unhealthy` | N failures → callback(service_id, False) | SEC-H5 |
| `test_recovery_after_unhealthy` | M successes after unhealthy → callback(service_id, True) | SEC-H5 |
| `test_backoff_on_failures` | Interval increases exponentially after failures | SEC-H3 |
| `test_backoff_resets_on_recovery` | Interval returns to default after recovery | |
| `test_jitter_applied` | Actual sleep time varies from configured interval | SEC-H3 |
| `test_check_timeout_prevents_hang` | Slow health_check() times out → treated as failure | SEC-H4 |
| `test_min_interval_enforced` | Interval below minimum raises ValueError | SEC-H3 |
| `test_get_health_status_returns_info` | Returns ServiceHealthInfo snapshot | |
| `test_get_health_status_returns_none_for_unmonitored` | Returns None | |
| `test_monitored_services_property` | Returns correct set of IDs | |

### 7.2 Mocking Pattern

```python
@pytest.fixture
def mock_registry() -> MagicMock:
    registry = MagicMock(spec=ServiceRegistry)
    registry.get = AsyncMock()
    return registry

@pytest.fixture
def mock_driver() -> MagicMock:
    driver = MagicMock(spec=ServiceDriver)
    driver.health_check = AsyncMock(return_value=True)
    return driver

@pytest.fixture
def monitor(mock_registry: MagicMock) -> HealthMonitor:
    return HealthMonitor(
        registry=mock_registry,
        default_interval=0.1,  # fast for tests
        failure_threshold=3,
        recovery_threshold=2,
        check_timeout=1.0,
        min_interval=0.05,  # allow fast intervals in tests
    )
```

---

## 8. File Locations

| File | Purpose |
|------|---------|
| `src/gpumod/services/health.py` | HealthMonitor implementation |
| `tests/unit/test_health_monitor.py` | Unit tests |
| `src/gpumod/services/__init__.py` | Re-export HealthMonitor |

---

## 9. Out of Scope (for P7-T3 implementation)

- Auto-restart on unhealthy (future: configurable restart policy per service).
- Health check history/metrics storage in DB.
- Alerting or notification system.
- Custom health check strategies (TCP, gRPC) — currently HTTP-only via driver ABC.
- Health dashboard in TUI (depends on P7-T4: Interactive TUI).
- Distributed health monitoring (multi-node).
