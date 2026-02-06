"""Jinja2 template rendering engine for gpumod systemd units.

Uses SandboxedEnvironment for security and StrictUndefined to catch
missing variables at render time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

if TYPE_CHECKING:
    from gpumod.models import Service

# Map driver types to their corresponding template file names.
_DRIVER_TEMPLATE_MAP: dict[str, str] = {
    "vllm": "vllm.service.j2",
    "llamacpp": "llamacpp.service.j2",
    "fastapi": "fastapi.service.j2",
}

# Pattern for validating template names: only allow simple filenames.
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_.]+$")

_TEMPLATES_DIR = Path(__file__).parent / "systemd"


class TemplateEngine:
    """Renders Jinja2 templates for systemd service units.

    Security:
    - Uses ``SandboxedEnvironment`` to prevent template injection attacks.
    - Uses ``StrictUndefined`` to fail on missing variables.
    - Validates template names against path traversal.
    """

    def __init__(self) -> None:
        """Initialize the engine with the built-in templates directory."""
        self._templates_dir = _TEMPLATES_DIR
        self._env = SandboxedEnvironment(
            loader=self._make_loader(),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def _make_loader(self) -> Any:
        """Create a FileSystemLoader for the templates directory."""
        from jinja2 import FileSystemLoader

        return FileSystemLoader(str(self._templates_dir))

    def _validate_template_name(self, template_name: str) -> None:
        """Validate that a template name is safe (no path traversal).

        Raises:
            ValueError: If the template name contains path traversal or
                is an absolute path.
        """
        if ".." in template_name:
            msg = f"Invalid template name (path traversal detected): {template_name}"
            raise ValueError(msg)
        if template_name.startswith("/"):
            msg = f"Invalid template name (absolute path not allowed): {template_name}"
            raise ValueError(msg)
        if not _SAFE_NAME_RE.match(template_name):
            msg = f"Invalid template name: {template_name}"
            raise ValueError(msg)

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a named template with the given context.

        Args:
            template_name: Name of the template file (e.g. ``vllm.service.j2``).
            context: Variables to pass to the template.

        Returns:
            The rendered template string.

        Raises:
            ValueError: If the template name is invalid.
            jinja2.TemplateNotFound: If the template does not exist.
            jinja2.UndefinedError: If a required variable is missing.
        """
        self._validate_template_name(template_name)
        template = self._env.get_template(template_name)
        return template.render(**context)

    def render_string(self, template_str: str, context: dict[str, Any]) -> str:
        """Render a template from a string.

        This is useful for user-supplied or database-stored templates.

        Args:
            template_str: The Jinja2 template as a string.
            context: Variables to pass to the template.

        Returns:
            The rendered template string.
        """
        template = self._env.from_string(template_str)
        return template.render(**context)

    def render_service_unit(
        self,
        service: Service,
        settings: dict[str, str],
        unit_vars: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Render the appropriate systemd unit file for a service.

        Selects the template based on ``service.driver``:
        - ``vllm`` -> ``vllm.service.j2``
        - ``llamacpp`` -> ``llamacpp.service.j2``
        - ``fastapi`` -> ``fastapi.service.j2``

        Args:
            service: The service definition.
            settings: System-level settings (user, cuda_devices, etc.).
            unit_vars: Driver-specific variables for the template.
            extra_env: Additional environment variables.

        Returns:
            The rendered systemd unit file content.

        Raises:
            ValueError: If no template exists for the service's driver type.
        """
        driver_key = str(service.driver)
        template_name = _DRIVER_TEMPLATE_MAP.get(driver_key)
        if template_name is None:
            msg = f"No template available for driver type: {service.driver}"
            raise ValueError(msg)

        context: dict[str, Any] = {
            "service": service,
            "settings": settings,
            "unit_vars": unit_vars if unit_vars is not None else {},
            "extra_env": extra_env if extra_env is not None else {},
        }
        return self.render(template_name, context)

    def available_templates(self) -> list[str]:
        """List available template names.

        Returns:
            Sorted list of template file names in the templates directory.
        """
        return sorted(
            p.name for p in self._templates_dir.iterdir() if p.is_file() and p.suffix == ".j2"
        )
