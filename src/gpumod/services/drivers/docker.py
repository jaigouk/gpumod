"""DockerDriver — service driver for Docker-containerized services.

Manages container lifecycle via the Docker SDK for Python. Supports
Qdrant, Langfuse, and other containerized services alongside
systemd-managed ML inference services.

Security: SEC-D7 (image validation), SEC-D8 (volume path traversal),
SEC-D9 (no privileged mode), SEC-D10 (name/env sanitization).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import docker
import docker.errors
import httpx

from gpumod.models import ServiceState, ServiceStatus
from gpumod.services.base import ServiceDriver
from gpumod.validation import (
    validate_container_runtime,
    validate_docker_env,
    validate_docker_image,
    validate_volume_mounts,
)

if TYPE_CHECKING:
    from gpumod.models import Service

logger = logging.getLogger(__name__)


class DockerDriver(ServiceDriver):
    """Driver for Docker-containerized services.

    Manages container lifecycle via the Docker SDK. Does not support
    sleep/wake — containers use stop/start for memory release.

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
    ) -> None:
        self._client = client or docker.from_env()
        self._http_timeout = http_timeout
        self._prefix = container_prefix

    # ------------------------------------------------------------------
    # ServiceDriver interface
    # ------------------------------------------------------------------

    async def start(self, service: Service) -> None:
        """Start a Docker container for the service.

        Validates image name, volume mounts, env vars, and runtime before
        creating the container. Hardcodes privileged=False (SEC-D9).
        """
        extra = service.extra_config
        image = self._require_image(extra)
        validate_docker_image(image)

        run_kwargs: dict[str, Any] = {
            "image": image,
            "name": self._container_name(service),
            "detach": True,
            "privileged": False,
        }

        if "volumes" in extra:
            validated = validate_volume_mounts(dict(extra["volumes"]))
            run_kwargs["volumes"] = validated

        if "environment" in extra:
            validated_env = validate_docker_env(dict(extra["environment"]))
            run_kwargs["environment"] = validated_env

        if "ports" in extra:
            run_kwargs["ports"] = extra["ports"]

        if "command" in extra:
            run_kwargs["command"] = extra["command"]

        if "runtime" in extra:
            validate_container_runtime(str(extra["runtime"]))
            run_kwargs["runtime"] = extra["runtime"]

        if "mem_limit" in extra:
            run_kwargs["mem_limit"] = extra["mem_limit"]

        await asyncio.to_thread(lambda: self._client.containers.run(**run_kwargs))
        logger.info("Started container %s", run_kwargs["name"])

    async def stop(self, service: Service) -> None:
        """Stop and remove a Docker container.

        Handles NotFound gracefully (container already gone).
        """
        name = self._container_name(service)
        try:
            container = await asyncio.to_thread(self._client.containers.get, name)
            await asyncio.to_thread(container.stop, timeout=30)
            await asyncio.to_thread(container.remove)
            logger.info("Stopped and removed container %s", name)
        except docker.errors.NotFound:
            logger.info("Container %s already gone", name)

    async def status(self, service: Service) -> ServiceStatus:
        """Determine the current service state from Docker container status."""
        name = self._container_name(service)
        try:
            container = await asyncio.to_thread(self._client.containers.get, name)
            docker_status: str = container.status

            if docker_status == "restarting":
                return ServiceStatus(state=ServiceState.STARTING)

            if docker_status in ("exited", "created"):
                return ServiceStatus(state=ServiceState.STOPPED)

            if docker_status == "running":
                healthy = await self.health_check(service)
                if healthy:
                    return ServiceStatus(state=ServiceState.RUNNING, health_ok=True)
                return ServiceStatus(state=ServiceState.UNHEALTHY, health_ok=False)

            return ServiceStatus(state=ServiceState.UNKNOWN)

        except docker.errors.NotFound:
            return ServiceStatus(state=ServiceState.STOPPED)
        except Exception:
            return ServiceStatus(state=ServiceState.UNKNOWN)

    async def health_check(self, service: Service) -> bool:
        """Check if the service is healthy via HTTP GET to the health endpoint."""
        if service.port is None:
            return False
        endpoint = service.health_endpoint or "/health"
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                resp = await client.get(f"http://localhost:{service.port}{endpoint}")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    @property
    def supports_sleep(self) -> bool:
        """Docker containers do not support sleep/wake."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _container_name(self, service: Service) -> str:
        """Build the container name from prefix and service ID."""
        return f"{self._prefix}-{service.id}"

    @staticmethod
    def _require_image(extra: dict[str, Any]) -> str:
        """Extract and validate that 'image' key exists in extra_config."""
        image = extra.get("image")
        if not image:
            msg = "Docker service requires 'image' in extra_config"
            raise ValueError(msg)
        return str(image)
