"""Unit tests for DockerDriver (P7-T2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gpumod.models import DriverType, Service, ServiceState


def make_docker_service(
    *,
    id: str = "test-docker",
    name: str = "Test Docker",
    port: int | None = 8080,
    vram_mb: int = 0,
    extra_config: dict[str, object] | None = None,
) -> Service:
    """Create a Service configured for Docker testing."""
    if extra_config is None:
        extra_config = {
            "image": "qdrant/qdrant:v1.8.0",
            "ports": {"6333": 8080},
        }
    return Service(
        id=id,
        name=name,
        driver=DriverType.DOCKER,
        vram_mb=vram_mb,
        port=port,
        extra_config=extra_config,
    )


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Create a mock Docker client."""
    client = MagicMock()
    client.containers = MagicMock()
    return client


@pytest.fixture
def driver(mock_docker_client: MagicMock) -> object:
    """Create a DockerDriver with a mocked Docker client."""
    from gpumod.services.drivers.docker import DockerDriver

    return DockerDriver(client=mock_docker_client)


# ---------------------------------------------------------------------------
# start() tests
# ---------------------------------------------------------------------------


class TestDockerStart:
    """Tests for DockerDriver.start()."""

    async def test_start_runs_container(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_docker_client.containers.run = MagicMock(return_value=MagicMock())

        await driver.start(svc)

        mock_docker_client.containers.run.assert_called_once()
        call_kwargs = mock_docker_client.containers.run.call_args
        assert call_kwargs.kwargs.get("detach") is True
        assert call_kwargs.kwargs.get("privileged") is not True

    async def test_start_rejects_invalid_image_name(self, driver: object) -> None:
        """SEC-D7: Image name injection blocked."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(extra_config={"image": "image; rm -rf /", "ports": {"80": 8080}})

        with pytest.raises(ValueError, match="[Ii]mage"):
            await driver.start(svc)

    async def test_start_rejects_path_traversal_in_volume(self, driver: object) -> None:
        """SEC-D8: Volume mount path traversal blocked."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(
            extra_config={
                "image": "qdrant/qdrant:v1.8.0",
                "volumes": {"../../etc/shadow": "/data"},
            }
        )

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            await driver.start(svc)

    async def test_start_no_privileged_mode(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        """SEC-D9: Containers never run with --privileged."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()
        mock_docker_client.containers.run = MagicMock(return_value=MagicMock())

        await driver.start(svc)

        call_kwargs = mock_docker_client.containers.run.call_args
        assert call_kwargs.kwargs.get("privileged") is False

    async def test_start_container_name_sanitized(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        """SEC-D10: Container name uses sanitized service.id with prefix."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(id="my-svc-1")
        mock_docker_client.containers.run = MagicMock(return_value=MagicMock())

        await driver.start(svc)

        call_kwargs = mock_docker_client.containers.run.call_args
        container_name = call_kwargs.kwargs.get("name")
        assert container_name == "gpumod-my-svc-1"

    async def test_start_missing_image_raises(self, driver: object) -> None:
        """Start with missing 'image' key in extra_config raises ValueError."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(extra_config={"ports": {"80": 8080}})

        with pytest.raises(ValueError, match="image"):
            await driver.start(svc)

    async def test_start_validates_env_keys(self, driver: object) -> None:
        """SEC-D10: Environment variable keys must be uppercase alphanumeric."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(
            extra_config={
                "image": "qdrant/qdrant:v1.8.0",
                "environment": {"valid_KEY": "ok", "bad;key": "evil"},
            }
        )

        with pytest.raises(ValueError, match="[Ee]nvironment|[Ee]nv"):
            await driver.start(svc)

    async def test_start_validates_runtime_allowlist(self, driver: object) -> None:
        """SEC-D9: Only 'runc' and 'nvidia' runtimes allowed."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(
            extra_config={
                "image": "qdrant/qdrant:v1.8.0",
                "runtime": "sysbox",
            }
        )

        with pytest.raises(ValueError, match="[Rr]untime"):
            await driver.start(svc)

    async def test_start_accepts_nvidia_runtime(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        """nvidia runtime is in the allowlist."""
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(
            extra_config={
                "image": "qdrant/qdrant:v1.8.0",
                "runtime": "nvidia",
            }
        )
        mock_docker_client.containers.run = MagicMock(return_value=MagicMock())

        await driver.start(svc)

        call_kwargs = mock_docker_client.containers.run.call_args
        assert call_kwargs.kwargs.get("runtime") == "nvidia"


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------


class TestDockerStop:
    """Tests for DockerDriver.stop()."""

    async def test_stop_removes_container(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_container = MagicMock()
        mock_docker_client.containers.get = MagicMock(return_value=mock_container)

        await driver.stop(svc)

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    async def test_stop_handles_not_found(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        """No error when container already gone."""
        import docker.errors

        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_docker_client.containers.get = MagicMock(
            side_effect=docker.errors.NotFound("Container not found")
        )

        await driver.stop(svc)


# ---------------------------------------------------------------------------
# status() tests
# ---------------------------------------------------------------------------


class TestDockerStatus:
    """Tests for DockerDriver.status()."""

    async def test_status_returns_running(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker_client.containers.get = MagicMock(return_value=mock_container)

        with patch.object(driver, "health_check", return_value=True):
            result = await driver.status(svc)

        assert result.state == ServiceState.RUNNING

    async def test_status_returns_stopped_when_exited(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_docker_client.containers.get = MagicMock(return_value=mock_container)

        result = await driver.status(svc)
        assert result.state == ServiceState.STOPPED

    async def test_status_returns_stopped_when_not_found(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        import docker.errors

        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_docker_client.containers.get = MagicMock(
            side_effect=docker.errors.NotFound("not found")
        )

        result = await driver.status(svc)
        assert result.state == ServiceState.STOPPED

    async def test_status_returns_unhealthy_when_running_but_unhealthy(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker_client.containers.get = MagicMock(return_value=mock_container)

        with patch.object(driver, "health_check", return_value=False):
            result = await driver.status(svc)

        assert result.state == ServiceState.UNHEALTHY

    async def test_status_returns_starting_when_restarting(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_container = MagicMock()
        mock_container.status = "restarting"
        mock_docker_client.containers.get = MagicMock(return_value=mock_container)

        result = await driver.status(svc)
        assert result.state == ServiceState.STARTING

    async def test_status_returns_unknown_on_error(
        self, driver: object, mock_docker_client: MagicMock
    ) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_docker_client.containers.get = MagicMock(side_effect=RuntimeError("unexpected error"))

        result = await driver.status(svc)
        assert result.state == ServiceState.UNKNOWN


# ---------------------------------------------------------------------------
# health_check() tests
# ---------------------------------------------------------------------------


class TestDockerHealthCheck:
    """Tests for DockerDriver.health_check()."""

    async def test_health_check_returns_true_on_200(self, driver: object) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("gpumod.services.drivers.docker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is True

    async def test_health_check_returns_false_on_connect_error(self, driver: object) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service()

        with patch("gpumod.services.drivers.docker.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await driver.health_check(svc)
            assert result is False

    async def test_health_check_returns_false_when_port_none(self, driver: object) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        svc = make_docker_service(port=None)

        result = await driver.health_check(svc)
        assert result is False


# ---------------------------------------------------------------------------
# supports_sleep property
# ---------------------------------------------------------------------------


class TestDockerSupportsSleep:
    """Tests for DockerDriver.supports_sleep property."""

    def test_supports_sleep_returns_false(self, driver: object) -> None:
        from gpumod.services.drivers.docker import DockerDriver

        assert isinstance(driver, DockerDriver)
        assert driver.supports_sleep is False


# ---------------------------------------------------------------------------
# Validation function tests
# ---------------------------------------------------------------------------


class TestDockerValidation:
    """Tests for Docker-specific validation functions."""

    def test_validate_docker_image_accepts_valid(self) -> None:
        from gpumod.validation import validate_docker_image

        assert validate_docker_image("qdrant/qdrant:v1.8.0") == "qdrant/qdrant:v1.8.0"
        assert validate_docker_image("nginx") == "nginx"
        assert validate_docker_image("registry.example.com/myapp:latest") == (
            "registry.example.com/myapp:latest"
        )

    def test_validate_docker_image_rejects_injection(self) -> None:
        from gpumod.validation import validate_docker_image

        with pytest.raises(ValueError, match="[Ii]mage"):
            validate_docker_image("image; rm -rf /")

    def test_validate_docker_image_rejects_empty(self) -> None:
        from gpumod.validation import validate_docker_image

        with pytest.raises(ValueError, match="[Ii]mage"):
            validate_docker_image("")

    def test_validate_volume_mounts_rejects_traversal(self) -> None:
        from gpumod.validation import validate_volume_mounts

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            validate_volume_mounts({"../../etc/shadow": "/data"})

    def test_validate_volume_mounts_rejects_etc(self) -> None:
        from gpumod.validation import validate_volume_mounts

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            validate_volume_mounts({"/etc/passwd": "/data"})

    def test_validate_docker_env_rejects_bad_keys(self) -> None:
        from gpumod.validation import validate_docker_env

        with pytest.raises(ValueError, match="[Ee]nv"):
            validate_docker_env({"bad;key": "value"})

    def test_validate_docker_env_accepts_valid(self) -> None:
        from gpumod.validation import validate_docker_env

        result = validate_docker_env({"MY_VAR": "hello", "PORT": "8080"})
        assert result == {"MY_VAR": "hello", "PORT": "8080"}

    def test_validate_container_runtime_rejects_unknown(self) -> None:
        from gpumod.validation import validate_container_runtime

        with pytest.raises(ValueError, match="[Rr]untime"):
            validate_container_runtime("sysbox")

    def test_validate_container_runtime_accepts_nvidia(self) -> None:
        from gpumod.validation import validate_container_runtime

        assert validate_container_runtime("nvidia") == "nvidia"

    def test_validate_container_runtime_accepts_runc(self) -> None:
        from gpumod.validation import validate_container_runtime

        assert validate_container_runtime("runc") == "runc"


# ---------------------------------------------------------------------------
# ServiceRegistry integration
# ---------------------------------------------------------------------------


class TestDockerRegistryIntegration:
    """Verify DockerDriver is registered in ServiceRegistry."""

    def test_registry_has_docker_driver(self) -> None:
        from gpumod.services.drivers.docker import DockerDriver
        from gpumod.services.registry import ServiceRegistry

        mock_db = MagicMock()
        registry = ServiceRegistry(db=mock_db)
        driver = registry.get_driver(DriverType.DOCKER)
        assert isinstance(driver, DockerDriver)
