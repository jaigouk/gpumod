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
        assert "WantedBy=default.target" in result

    def test_no_user_directive_in_user_units(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        """User-level systemd units must not contain User= directive."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "User=" not in result

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

    def test_dtype_rendered(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"dtype": "float16"}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--dtype float16" in result

    def test_enforce_eager(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"enforce_eager": True}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--enforce-eager" in result

    def test_enforce_eager_false_omitted(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        result = engine.render_service_unit(vllm_service, default_settings)
        assert "--enforce-eager" not in result

    def test_runner_pooling_maps_to_embed(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        """runner='pooling' should map to --task embed for vllm."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"runner": "pooling"}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--task embed" in result
        assert "--task pooling" not in result

    def test_runner_generate(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"runner": "generate"}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--task generate" in result

    def test_hf_overrides(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        overrides = '{"architectures": ["CustomArch"]}'
        unit_vars: dict[str, Any] = {"hf_overrides": overrides}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert f"--hf-overrides '{overrides}'" in result

    def test_sleep_mode_flags(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"enable_sleep_mode": True, "sleep_level": 2}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--enable-sleep-mode" in result
        assert "--sleep-level 2" in result

    def test_trust_remote_code(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"trust_remote_code": True}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--trust-remote-code" in result

    def test_max_num_seqs(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"max_num_seqs": 8}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--max-num-seqs 8" in result

    def test_extra_args(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {"extra_args": "--quantization awq"}
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--quantization awq" in result

    def test_full_reranker_preset(
        self, vllm_service: Service, default_settings: dict[str, str]
    ) -> None:
        """Full reranker-like preset renders all flags correctly."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "gpu_mem_util": 0.25,
            "max_model_len": 512,
            "max_num_seqs": 8,
            "dtype": "float16",
            "enforce_eager": True,
            "runner": "pooling",
            "hf_overrides": '{"architectures": ["Qwen3VLForSequenceClassification"]}',
            "enable_sleep_mode": True,
            "sleep_level": 2,
        }
        result = engine.render_service_unit(vllm_service, default_settings, unit_vars=unit_vars)
        assert "--gpu-memory-utilization 0.25" in result
        assert "--max-model-len 512" in result
        assert "--max-num-seqs 8" in result
        assert "--dtype float16" in result
        assert "--enforce-eager" in result
        assert "--task embed" in result  # pooling -> embed
        assert "--hf-overrides" in result
        assert "--enable-sleep-mode" in result
        assert "--sleep-level 2" in result


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

    def test_router_mode_models_dir(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        """Router mode uses --models-dir instead of --model."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "models_dir": "/home/user/bin",
            "no_models_autoload": True,
            "models_max": 1,
            "jinja": True,
        }
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--models-dir /home/user/bin" in result
        assert "--no-models-autoload" in result
        assert "--models-max 1" in result
        assert "--jinja" in result
        assert "--model " not in result  # no --model in router mode

    def test_router_mode_without_models_max(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        """Router mode without models_max omits --models-max flag."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "models_dir": "/home/user/bin",
            "no_models_autoload": True,
        }
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--models-dir /home/user/bin" in result
        assert "--no-models-autoload" in result
        assert "--models-max" not in result

    def test_router_mode_omits_ctx_size(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        """Router mode omits --ctx-size unless explicitly set."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "models_dir": "/home/user/bin",
        }
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--ctx-size" not in result

    def test_router_mode_explicit_ctx_size(
        self, llamacpp_service: Service, default_settings: dict[str, str]
    ) -> None:
        """Router mode includes --ctx-size when explicitly set."""
        from gpumod.templates.engine import TemplateEngine

        engine = TemplateEngine()
        unit_vars: dict[str, Any] = {
            "models_dir": "/home/user/bin",
            "context_size": 8192,
        }
        result = engine.render_service_unit(
            llamacpp_service, default_settings, unit_vars=unit_vars
        )
        assert "--ctx-size 8192" in result


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
