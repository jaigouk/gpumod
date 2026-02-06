"""Tests for the gpumod template engine and Jinja2 systemd unit templates."""

from __future__ import annotations

from typing import Any

import pytest
from jinja2 import TemplateNotFound, UndefinedError
from jinja2.exceptions import SecurityError

from gpumod.models import DriverType, Service

# ── Helper fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def vllm_service() -> Service:
    """A vLLM service for testing."""
    return Service(
        id="vllm-chat",
        name="vLLM Chat Service",
        driver=DriverType.VLLM,
        port=8000,
        vram_mb=8000,
        model_id="mistralai/Devstral-Small-2505",
    )


@pytest.fixture
def llamacpp_service() -> Service:
    """A llama.cpp service for testing."""
    return Service(
        id="llama-code",
        name="llama.cpp Code",
        driver=DriverType.LLAMACPP,
        port=8080,
        vram_mb=6000,
    )


@pytest.fixture
def fastapi_service() -> Service:
    """A FastAPI service for testing."""
    return Service(
        id="proxy-api",
        name="FastAPI Proxy",
        driver=DriverType.FASTAPI,
        port=9000,
        vram_mb=0,
    )


@pytest.fixture
def default_settings() -> dict[str, str]:
    """Default settings dict for rendering."""
    return {
        "user": "gpumod",
        "cuda_devices": "0",
        "hf_home": "/data/.cache/huggingface",
    }


# ── TemplateEngine initialization ───────────────────────────────────────


class TestTemplateEngineInit:
    def test_creates_sandboxed_environment(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        # Should use SandboxedEnvironment internally
        assert engine is not None

    def test_strict_undefined(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        # Rendering a template with missing required variables should raise
        with pytest.raises(UndefinedError):
            engine.render_string("{{ missing_var }}", {})


# ── available_templates ─────────────────────────────────────────────────


class TestAvailableTemplates:
    def test_lists_builtin_templates(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        templates = engine.available_templates()
        assert "vllm.service.j2" in templates
        assert "llamacpp.service.j2" in templates
        assert "fastapi.service.j2" in templates

    def test_returns_exactly_three_templates(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        templates = engine.available_templates()
        assert len(templates) == 3


# ── render_service_unit for vllm ────────────────────────────────────────


class TestRenderVllmUnit:
    def test_contains_unit_section(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "[Unit]" in result
        assert "Description=vLLM Chat Service" in result

    def test_contains_service_section(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "[Service]" in result
        assert "Type=simple" in result

    def test_contains_exec_start(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "ExecStart=" in result
        assert "vllm" in result.lower() or "serve" in result
        assert "mistralai/Devstral-Small-2505" in result
        assert "--port 8000" in result

    def test_contains_install_section(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "[Install]" in result
        assert "WantedBy=multi-user.target" in result

    def test_user_from_settings(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "User=gpumod" in result

    def test_cuda_visible_devices(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "CUDA_VISIBLE_DEVICES=0" in result

    def test_unit_vars_override(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "gpu_mem_util": 0.85,
            "max_model_len": 8192,
        }
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "0.85" in result
        assert "8192" in result

    def test_extra_env_variables(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        extra_env = {"NCCL_P2P_DISABLE": "1", "MY_VAR": "hello"}
        result = engine.render_service_unit(vllm_service, default_settings, extra_env=extra_env)
        assert 'Environment="NCCL_P2P_DISABLE=1"' in result
        assert 'Environment="MY_VAR=hello"' in result

    def test_hf_home_variable(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "HF_HOME=/data/.cache/huggingface" in result


# ── render_service_unit for llamacpp ────────────────────────────────────


class TestRenderLlamacppUnit:
    def test_contains_llamacpp_exec(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"model_path": "/models/code.gguf"}
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "llama-server" in result or "llamacpp" in result.lower()
        assert "--model /models/code.gguf" in result
        assert "--port 8080" in result

    def test_context_size_default(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"model_path": "/models/code.gguf"}
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--ctx-size 4096" in result

    def test_gpu_layers_default(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"model_path": "/models/code.gguf"}
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--n-gpu-layers -1" in result

    def test_flash_attn_enabled(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "model_path": "/models/code.gguf",
            "flash_attn": True,
        }
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--flash-attn" in result

    def test_flash_attn_disabled_by_default(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"model_path": "/models/code.gguf"}
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--flash-attn" not in result


# ── render_service_unit for fastapi ─────────────────────────────────────


class TestRenderFastapiUnit:
    def test_contains_uvicorn_exec(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(fastapi_service, default_settings)
        assert "uvicorn" in result
        assert "--port 9000" in result

    def test_app_module_default(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(fastapi_service, default_settings)
        assert "main:app" in result

    def test_custom_app_module(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"app_module": "myapp.server:application"}
        result = engine.render_service_unit(fastapi_service, default_settings, unit_vars=unit_vars)
        assert "myapp.server:application" in result

    def test_working_directory(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"working_dir": "/srv/myapp"}
        result = engine.render_service_unit(fastapi_service, default_settings, unit_vars=unit_vars)
        assert "WorkingDirectory=/srv/myapp" in result

    def test_workers_option(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"workers": 4}
        result = engine.render_service_unit(fastapi_service, default_settings, unit_vars=unit_vars)
        assert "--workers 4" in result


# ── Driver-based template selection ─────────────────────────────────────


class TestDriverTemplateSelection:
    def test_vllm_driver_uses_vllm_template(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        # Should contain vllm-specific content
        assert "serve" in result

    def test_llamacpp_driver_uses_llamacpp_template(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"model_path": "/models/test.gguf"}
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--model" in result

    def test_fastapi_driver_uses_fastapi_template(
        self, fastapi_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(fastapi_service, default_settings)
        assert "uvicorn" in result

    def test_unsupported_driver_raises(self, default_settings: dict[str, str]) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        svc = Service(
            id="docker-svc",
            name="Docker Service",
            driver=DriverType.DOCKER,
            vram_mb=4096,
            port=5000,
        )
        with pytest.raises(ValueError, match="[Nn]o template"):
            engine.render_service_unit(svc, default_settings)


# ── render and render_string ────────────────────────────────────────────


class TestRenderMethods:
    def test_render_named_template(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        svc = Service(
            id="test-svc",
            name="Test",
            driver=DriverType.VLLM,
            port=8000,
            vram_mb=4096,
            model_id="test/model",
        )
        ctx: dict[str, Any] = {
            "service": svc,
            "settings": {"user": "root"},
            "unit_vars": {},
            "extra_env": {},
        }
        result = engine.render("vllm.service.j2", ctx)
        assert "[Unit]" in result
        assert "Description=Test" in result

    def test_render_unknown_template_raises(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        with pytest.raises(TemplateNotFound):
            engine.render("nonexistent.j2", {})

    def test_render_string_basic(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_string("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_string_with_conditionals(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        tpl = "{% if enabled %}ON{% else %}OFF{% endif %}"
        assert engine.render_string(tpl, {"enabled": True}) == "ON"
        assert engine.render_string(tpl, {"enabled": False}) == "OFF"


# ── Security tests ──────────────────────────────────────────────────────


class TestTemplateSecurity:
    def test_path_traversal_rejected(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        with pytest.raises(ValueError, match="[Ii]nvalid|traversal|[Pp]ath"):
            engine.render("../../../etc/passwd", {})

    def test_absolute_path_rejected(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        with pytest.raises(ValueError, match="[Ii]nvalid|traversal|[Pp]ath"):
            engine.render("/etc/passwd", {})

    def test_sandbox_prevents_file_access(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        # Attempting to access file system from template should fail
        malicious = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        with pytest.raises(SecurityError):
            engine.render_string(malicious, {})

    def test_sandbox_prevents_import(self) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        # Attempting to use __import__ should fail in sandbox
        malicious = "{% for c in [].__class__.__bases__[0].__subclasses__() %}{{ c }}{% endfor %}"
        with pytest.raises(SecurityError):
            engine.render_string(malicious, {})
