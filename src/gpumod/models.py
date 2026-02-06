"""Pydantic models and enums for gpumod."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ServiceState(StrEnum):
    """Possible states of a managed GPU service."""

    UNKNOWN = "unknown"
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    SLEEPING = "sleeping"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    FAILED = "failed"


class DriverType(StrEnum):
    """Service driver types."""

    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    FASTAPI = "fastapi"
    DOCKER = "docker"


class SleepMode(StrEnum):
    """Sleep mode levels for GPU memory optimization."""

    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ROUTER = "router"


class ServiceStatus(BaseModel):
    """Runtime status of a service."""

    model_config = ConfigDict(extra="forbid")

    state: ServiceState
    vram_mb: int | None = None
    uptime_seconds: int | None = None
    health_ok: bool | None = None
    sleep_level: str | None = None
    last_error: str | None = None


class Service(BaseModel):
    """Definition of a managed GPU service."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    driver: DriverType
    port: int | None = None
    vram_mb: int
    sleep_mode: SleepMode = SleepMode.NONE
    health_endpoint: str = "/health"
    model_id: str | None = None
    unit_name: str | None = None
    depends_on: list[str] = []
    startup_timeout: int = 120
    extra_config: dict[str, Any] = {}


class Mode(BaseModel):
    """A named collection of services for a specific use case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str | None = None
    services: list[str] = []
    total_vram_mb: int | None = None


class ModeResult(BaseModel):
    """Result of a mode switch operation."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    mode_id: str
    started: list[str] = []
    stopped: list[str] = []
    message: str | None = None
    errors: list[str] = []


class GPUInfo(BaseModel):
    """GPU hardware information."""

    model_config = ConfigDict(extra="forbid")

    name: str
    vram_total_mb: int
    driver: str | None = None
    architecture: str | None = None


class VRAMUsage(BaseModel):
    """Current GPU VRAM usage."""

    model_config = ConfigDict(extra="forbid")

    total_mb: int
    used_mb: int
    free_mb: int


class ServiceInfo(BaseModel):
    """Combined service definition and runtime status."""

    model_config = ConfigDict(extra="forbid")

    service: Service
    status: ServiceStatus


class SystemStatus(BaseModel):
    """Full system status including GPU, VRAM, and services."""

    model_config = ConfigDict(extra="forbid")

    gpu: GPUInfo | None = None
    vram: VRAMUsage | None = None
    current_mode: str | None = None
    services: list[ServiceInfo] = []
