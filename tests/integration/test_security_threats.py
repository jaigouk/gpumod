"""Integration tests covering all security threats (T1-T21).

Test matrix mapping each threat to its SEC control and test:

| Threat | SEC Control | Test Class |
|--------|-------------|------------|
| T1     | SEC-V1, SEC-D1 | TestShellInjectionViaIDs |
| T2     | SEC-V1, SEC-D2 | TestSQLInjection |
| T5     | SEC-E1 | TestErrorSanitization |
| T6     | SEC-E2 | TestResourceDisclosure |
| T7     | SEC-A1 | TestMutatingToolClassification |
| T8     | SEC-R1 | TestSimulationCap |
| T11    | SEC-V2 | TestExtraFieldsRejected |
| T13    | SEC-L4 | TestLLMResponseManipulation |
| T17    | SEC-L5 | TestDataMinimization |
| T18    | SEC-D7 | TestDockerImageInjection |
| T19    | SEC-D8 | TestDockerVolumeTraversal |
| T20    | SEC-D9 | TestDockerPrivileged |
| T21    | SEC-D10 | TestDockerEnvInjection |

Threats T3, T4, T9, T10, T12, T14, T15, T16 are covered in
test_security_integration.py and test_security_phase6.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from gpumod.db import Database
from gpumod.models import DriverType, Service, ServiceState

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# T1: Shell injection via service/mode IDs (SEC-V1, SEC-D1)
# ---------------------------------------------------------------------------


class TestShellInjectionViaIDs:
    """T1: Shell metacharacters in service_id/mode_id are rejected at MCP boundary."""

    def test_service_id_shell_injection_rejected(self) -> None:
        """service_id with shell injection is rejected by validate_service_id."""
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id('; rm -rf /"')

    def test_service_id_pipe_injection_rejected(self) -> None:
        """service_id with pipe operator is rejected."""
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("svc | cat /etc/passwd")

    def test_service_id_backtick_injection_rejected(self) -> None:
        """service_id with backtick command substitution is rejected."""
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("`whoami`")

    def test_mode_id_shell_injection_rejected(self) -> None:
        """mode_id with shell metacharacters is rejected."""
        from gpumod.validation import validate_mode_id

        with pytest.raises(ValueError, match="Invalid mode_id"):
            validate_mode_id("mode; shutdown -h now")

    def test_mcp_tool_rejects_shell_injection_in_service_id(self) -> None:
        """MCP tool boundary returns validation error for injected service_id."""
        from gpumod.validation import validate_service_id

        payloads = [
            "$(cat /etc/passwd)",
            "svc && rm -rf /",
            "svc\nnewline",
            'svc"injected',
        ]
        for payload in payloads:
            with pytest.raises(ValueError, match="Invalid service_id"):
                validate_service_id(payload)

    def test_systemd_unit_name_regex(self) -> None:
        """Systemd module only accepts safe unit names."""
        from gpumod.services.systemd import UNIT_NAME_PATTERN

        assert UNIT_NAME_PATTERN.match("valid-service.service")
        assert UNIT_NAME_PATTERN.match("my_svc.service")
        assert not UNIT_NAME_PATTERN.match("; rm -rf /.service")
        assert not UNIT_NAME_PATTERN.match("svc.service; echo pwned")


# ---------------------------------------------------------------------------
# T2: SQL injection via string args (SEC-V1, SEC-D2)
# ---------------------------------------------------------------------------


class TestSQLInjection:
    """T2: SQL injection via string args blocked by regex + parameterized queries."""

    def test_sql_injection_in_service_id_rejected(self) -> None:
        """service_id with SQL injection is rejected by validation."""
        from gpumod.validation import validate_service_id

        with pytest.raises(ValueError, match="Invalid service_id"):
            validate_service_id("'; DROP TABLE services--")

    async def test_parameterized_query_prevents_injection(self, tmp_path: Path) -> None:
        """DB uses parameterized queries; injected values don't execute as SQL."""
        db = Database(tmp_path / "sql_test.db")
        await db.connect()
        try:
            svc = Service(
                id="safe-svc",
                name="Normal Service",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
            )
            await db.insert_service(svc)

            result = await db.get_service("safe-svc' OR '1'='1")
            assert result is None

            all_svcs = await db.list_services()
            assert len(all_svcs) == 1
            assert all_svcs[0].id == "safe-svc"
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# T5: Information disclosure via errors (SEC-E1)
# ---------------------------------------------------------------------------


class TestErrorSanitization:
    """T5: Internal paths and tracebacks stripped from error messages."""

    def test_absolute_path_stripped(self) -> None:
        """Absolute file paths are replaced with [redacted]."""
        from gpumod.mcp_server import sanitize_error_message

        msg = "FileNotFoundError: /home/user/.config/gpumod/db.sqlite not found"
        sanitized = sanitize_error_message(msg)
        assert "/home/user" not in sanitized
        assert "gpumod" not in sanitized or "redacted" in sanitized

    def test_traceback_stripped(self) -> None:
        """Full Python tracebacks are replaced with generic message."""
        from gpumod.mcp_server import sanitize_error_message

        msg = (
            "Traceback (most recent call last):\n"
            '  File "/opt/gpumod/src/gpumod/mcp_tools.py", line 42, in service_info\n'
            "    result = await registry.get(service_id)\n"
            "KeyError: 'svc-1'"
        )
        sanitized = sanitize_error_message(msg)
        assert "Traceback" not in sanitized
        assert "/opt/gpumod" not in sanitized
        assert "mcp_tools.py" not in sanitized

    def test_file_line_reference_stripped(self) -> None:
        """File/line references are stripped."""
        from gpumod.mcp_server import sanitize_error_message

        msg = 'Error at File "src/gpumod/db.py", line 123'
        sanitized = sanitize_error_message(msg)
        assert "db.py" not in sanitized
        assert "line 123" not in sanitized

    def test_safe_message_unchanged(self) -> None:
        """Messages without sensitive data pass through."""
        from gpumod.mcp_server import sanitize_error_message

        msg = "Service not found: svc-1"
        sanitized = sanitize_error_message(msg)
        assert "Service not found" in sanitized


# ---------------------------------------------------------------------------
# T6: Information disclosure via resources (SEC-E2)
# ---------------------------------------------------------------------------


class TestResourceDisclosure:
    """T6: Resource output must not contain absolute paths or internal config."""

    async def test_resource_output_no_absolute_paths(self, tmp_path: Path) -> None:
        """MCP resource responses for services contain no absolute paths."""
        db = Database(tmp_path / "resource_test.db")
        await db.connect()
        try:
            svc = Service(
                id="test-svc",
                name="Test Service",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
                unit_name="test-svc.service",
            )
            await db.insert_service(svc)

            result = await db.get_service("test-svc")
            assert result is not None
            output = result.model_dump_json()
            assert "/home/" not in output
            assert "/opt/" not in output
            assert "/etc/" not in output
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# T7: Unauthorized mutating operations (SEC-A1)
# ---------------------------------------------------------------------------


class TestMutatingToolClassification:
    """T7: Mutating tools are labeled and non-read operations are identifiable."""

    def test_mutating_tools_marked_in_docstring(self) -> None:
        """All mutating MCP tools have [MUTATING] in their docstring."""
        from gpumod import mcp_tools

        mutating_tools = ["switch_mode", "start_service", "stop_service"]
        for tool_name in mutating_tools:
            func = getattr(mcp_tools, tool_name)
            assert func.__doc__ is not None
            assert "[MUTATING]" in func.__doc__, f"{tool_name} docstring missing [MUTATING] marker"

    def test_read_only_tools_not_marked_mutating(self) -> None:
        """Read-only MCP tools do NOT have [MUTATING] in their docstring."""
        from gpumod import mcp_tools

        read_only_tools = ["list_services", "service_info", "simulate_mode"]
        for tool_name in read_only_tools:
            func = getattr(mcp_tools, tool_name)
            assert func.__doc__ is not None
            assert "[MUTATING]" not in func.__doc__, f"{tool_name} should not be marked [MUTATING]"


# ---------------------------------------------------------------------------
# T8: Resource exhaustion via simulation (SEC-R1)
# ---------------------------------------------------------------------------


class TestSimulationCap:
    """T8: Simulation alternatives are capped at MAX_ALTERNATIVES (10)."""

    def test_max_alternatives_constant_exists(self) -> None:
        """_MAX_ALTERNATIVES is defined and equals 10."""
        from gpumod.simulation import _MAX_ALTERNATIVES

        assert _MAX_ALTERNATIVES == 10

    async def test_simulation_results_capped(self, tmp_path: Path) -> None:
        """SimulationEngine.generate_alternatives returns at most 10 results."""
        from gpumod.services.vram import VRAMTracker
        from gpumod.simulation import _MAX_ALTERNATIVES, SimulationEngine

        db = Database(tmp_path / "sim_test.db")
        await db.connect()
        try:
            from gpumod.registry import ModelRegistry

            model_reg = ModelRegistry(db)
            vram = VRAMTracker()
            engine = SimulationEngine(db=db, vram=vram, model_registry=model_reg)

            services = [
                Service(
                    id=f"svc-{i}",
                    name=f"Service {i}",
                    driver=DriverType.VLLM,
                    port=8000 + i,
                    vram_mb=2000,
                )
                for i in range(20)
            ]

            alternatives = await engine._generate_alternatives(
                services=services,
                total_vram=40000,
                gpu_total_mb=24000,
            )

            assert len(alternatives) <= _MAX_ALTERNATIVES
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# T11: Extra fields in tool input (SEC-V2)
# ---------------------------------------------------------------------------


class TestExtraFieldsRejected:
    """T11: Pydantic models with extra='forbid' reject unexpected fields."""

    def test_service_model_rejects_extra_fields(self) -> None:
        """Service model with extra='forbid' rejects unknown fields."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="extra"):
            Service(
                id="svc-1",
                name="Test",
                driver=DriverType.VLLM,
                port=8000,
                vram_mb=4000,
                unknown_field="hacker_value",
            )

    def test_service_status_rejects_extra_fields(self) -> None:
        """ServiceStatus model rejects extra fields."""
        from pydantic import ValidationError

        from gpumod.models import ServiceStatus

        with pytest.raises(ValidationError, match="extra"):
            ServiceStatus(
                state=ServiceState.RUNNING,
                injected_field="malicious",
            )

    def test_mcp_server_has_strict_input_validation(self) -> None:
        """MCP server is created with strict_input_validation=True."""
        from gpumod.mcp_server import create_mcp_server

        server = create_mcp_server()
        assert server.strict_input_validation is True


# ---------------------------------------------------------------------------
# T13: LLM response manipulation (SEC-L4)
# ---------------------------------------------------------------------------


class TestLLMResponseManipulation:
    """T13: LLM plans are advisory-only; never auto-executed."""

    def test_plan_response_is_data_only(self) -> None:
        """validate_plan_response returns data; no side effects or execution."""
        from gpumod.llm.response_validator import validate_plan_response

        raw: dict[str, Any] = {
            "services": [
                {"service_id": "svc-1", "vram_mb": 4000},
                {"service_id": "svc-2", "vram_mb": 8000},
            ],
            "reasoning": "Optimized allocation for inference.",
        }
        plan = validate_plan_response(raw)

        assert len(plan.services) == 2
        assert plan.reasoning == "Optimized allocation for inference."

    def test_plan_with_impossible_vram_still_parses(self) -> None:
        """Impossible VRAM plans parse but are advisory (validation catches it)."""
        from gpumod.llm.response_validator import validate_plan_response

        raw: dict[str, Any] = {
            "services": [
                {"service_id": "svc-1", "vram_mb": 999999},
            ],
            "reasoning": "Allocate everything.",
        }
        plan = validate_plan_response(raw)
        assert plan.services[0].vram_mb == 999999

    def test_cli_plan_does_not_auto_execute(self) -> None:
        """The plan command in CLI only suggests; it does not call switch_mode."""
        import inspect

        from gpumod import cli_plan

        source = inspect.getsource(cli_plan)
        assert "switch_mode" not in source, (
            "cli_plan should not call switch_mode (plans are advisory-only)"
        )


# ---------------------------------------------------------------------------
# T17: Sensitive data sent to external LLM APIs (SEC-L5)
# ---------------------------------------------------------------------------


class TestDataMinimization:
    """T17: Only service IDs, names, VRAM, and GPU capacity sent to LLM APIs."""

    def test_prompt_contains_only_minimal_fields(self) -> None:
        """build_planning_prompt only includes id, name, vram_mb in service data."""
        from gpumod.llm.prompts import build_planning_prompt

        services: list[dict[str, Any]] = [
            {
                "id": "svc-1",
                "name": "Test Service",
                "vram_mb": 4000,
                "port": 8000,
                "unit_name": "/etc/systemd/system/svc-1.service",
                "model_id": "org/secret-model",
                "extra_config": {"api_key": "sk-secret"},
            },
        ]
        prompt = build_planning_prompt(services, gpu_total_mb=24576)

        assert "svc-1" in prompt
        assert "Test Service" in prompt
        assert "4000" in prompt
        assert "24576" in prompt

        assert "8000" not in prompt
        assert "/etc/systemd" not in prompt
        assert "secret-model" not in prompt
        assert "sk-secret" not in prompt
        assert "api_key" not in prompt

    def test_prompt_data_boundary_markers(self) -> None:
        """Prompt has clear data boundaries to prevent injection."""
        from gpumod.llm.prompts import build_planning_prompt

        services: list[dict[str, Any]] = [
            {"id": "svc-1", "name": "Test", "vram_mb": 4000},
        ]
        prompt = build_planning_prompt(services, gpu_total_mb=24576)

        assert "--- BEGIN SERVICE DATA ---" in prompt
        assert "--- END SERVICE DATA ---" in prompt


# ---------------------------------------------------------------------------
# T18: Docker image name injection (SEC-D7)
# ---------------------------------------------------------------------------


class TestDockerImageInjection:
    """T18: Image name injection via Docker extra_config is blocked."""

    def test_image_shell_injection_rejected(self) -> None:
        """Image name with shell metacharacters is rejected."""
        from gpumod.validation import validate_docker_image

        with pytest.raises(ValueError, match="[Ii]mage"):
            validate_docker_image("image; rm -rf /")

    def test_image_command_substitution_rejected(self) -> None:
        from gpumod.validation import validate_docker_image

        with pytest.raises(ValueError, match="[Ii]mage"):
            validate_docker_image("$(malicious-command)")

    async def test_docker_driver_start_rejects_bad_image(self) -> None:
        """DockerDriver.start() rejects a service with an invalid image name."""
        from gpumod.services.drivers.docker import DockerDriver

        mock_client = MagicMock()
        driver = DockerDriver(client=mock_client)

        svc = Service(
            id="bad-image",
            name="Bad Image",
            driver=DriverType.DOCKER,
            port=8080,
            vram_mb=0,
            extra_config={"image": "UPPERCASE_NOT_ALLOWED:tag"},
        )
        with pytest.raises(ValueError, match="[Ii]mage"):
            await driver.start(svc)


# ---------------------------------------------------------------------------
# T19: Docker volume mount path traversal (SEC-D8)
# ---------------------------------------------------------------------------


class TestDockerVolumeTraversal:
    """T19: Volume mount path traversal is blocked."""

    def test_relative_path_rejected(self) -> None:
        from gpumod.validation import validate_volume_mounts

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            validate_volume_mounts({"../../../etc/shadow": "/data"})

    def test_etc_path_rejected(self) -> None:
        from gpumod.validation import validate_volume_mounts

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            validate_volume_mounts({"/etc/passwd": "/data"})

    def test_var_run_path_rejected(self) -> None:
        """Docker socket mount is rejected."""
        from gpumod.validation import validate_volume_mounts

        with pytest.raises(ValueError, match="[Vv]olume|[Pp]ath"):
            validate_volume_mounts({"/var/run/docker.sock": "/var/run/docker.sock"})

    def test_allowed_tmp_path_accepted(self) -> None:
        from gpumod.validation import validate_volume_mounts

        result = validate_volume_mounts({"/tmp/data": "/data"})
        assert result == {"/tmp/data": "/data"}


# ---------------------------------------------------------------------------
# T20: Docker privileged container escape (SEC-D9)
# ---------------------------------------------------------------------------


class TestDockerPrivileged:
    """T20: Containers never run with --privileged."""

    async def test_privileged_always_false_in_run_call(self) -> None:
        """DockerDriver always passes privileged=False to containers.run()."""
        from gpumod.services.drivers.docker import DockerDriver

        mock_client = MagicMock()
        mock_client.containers.run = MagicMock(return_value=MagicMock())
        driver = DockerDriver(client=mock_client)

        svc = Service(
            id="priv-test",
            name="Priv Test",
            driver=DriverType.DOCKER,
            port=8080,
            vram_mb=0,
            extra_config={"image": "nginx:1.25"},
        )
        await driver.start(svc)

        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs.kwargs["privileged"] is False

    def test_invalid_runtime_rejected(self) -> None:
        """Container runtime outside allowlist is rejected."""
        from gpumod.validation import validate_container_runtime

        with pytest.raises(ValueError, match="[Rr]untime"):
            validate_container_runtime("sysbox")

        with pytest.raises(ValueError, match="[Rr]untime"):
            validate_container_runtime("privileged")


# ---------------------------------------------------------------------------
# T21: Docker container name / env var injection (SEC-D10)
# ---------------------------------------------------------------------------


class TestDockerEnvInjection:
    """T21: Container name and env var injection blocked."""

    def test_env_key_with_special_chars_rejected(self) -> None:
        from gpumod.validation import validate_docker_env

        with pytest.raises(ValueError, match="[Ee]nv"):
            validate_docker_env({"bad;key": "value"})

    def test_env_key_with_lowercase_rejected(self) -> None:
        from gpumod.validation import validate_docker_env

        with pytest.raises(ValueError, match="[Ee]nv"):
            validate_docker_env({"lowercase_key": "value"})

    def test_env_value_with_newline_rejected(self) -> None:
        from gpumod.validation import validate_docker_env

        with pytest.raises(ValueError, match="[Ee]nv"):
            validate_docker_env({"VALID_KEY": "value\ninjected"})

    async def test_container_name_uses_prefix(self) -> None:
        """Container name uses gpumod- prefix with sanitized service ID."""
        from gpumod.services.drivers.docker import DockerDriver

        mock_client = MagicMock()
        mock_client.containers.run = MagicMock(return_value=MagicMock())
        driver = DockerDriver(client=mock_client, container_prefix="gpumod")

        svc = Service(
            id="my-svc",
            name="My Service",
            driver=DriverType.DOCKER,
            port=8080,
            vram_mb=0,
            extra_config={"image": "nginx:1.25"},
        )
        await driver.start(svc)

        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs.kwargs["name"] == "gpumod-my-svc"

    def test_valid_env_accepted(self) -> None:
        from gpumod.validation import validate_docker_env

        result = validate_docker_env({"MY_VAR": "hello", "DB_HOST": "localhost"})
        assert result == {"MY_VAR": "hello", "DB_HOST": "localhost"}
