"""UnitFileInstaller — auto-installs systemd user unit files before service start.

When the lifecycle manager starts a service, the unit file must exist under
``~/.config/systemd/user/``.  This class checks for the file and renders it
from the template engine if it is missing, removing the need for a manual
``gpumod template install`` step.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpumod.db import Database
    from gpumod.models import Service
    from gpumod.templates.engine import TemplateEngine

logger = logging.getLogger(__name__)

_SYSTEMD_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


async def _build_settings(db: Database) -> dict[str, str]:
    """Build a settings dict from DB settings for template rendering.

    Mirrors ``cli_template._build_settings`` so that auto-installed units
    get the same content as manually installed ones.
    """
    settings: dict[str, str] = {}
    for key in ("user", "cuda_devices", "working_dir"):
        val: str | None = await db.get_setting(key)
        if val is not None:
            settings[key] = val

    _binary_keys = {
        "vllm_bin": "vllm",
        "llamacpp_bin": "llama-server",
        "uvicorn_bin": "uvicorn",
    }
    for setting_key, binary_name in _binary_keys.items():
        val = await db.get_setting(setting_key)
        if val is None:
            resolved = shutil.which(binary_name)
            if resolved:
                settings[setting_key] = resolved
        else:
            settings[setting_key] = val

    return settings


class UnitFileInstaller:
    """Checks for and auto-installs systemd user unit files.

    Parameters
    ----------
    db:
        Database for reading settings.
    template_engine:
        Template engine for rendering unit files.
    unit_dir:
        Target directory for unit files.  Defaults to
        ``~/.config/systemd/user/``.
    """

    def __init__(
        self,
        db: Database,
        template_engine: TemplateEngine,
        *,
        unit_dir: Path | None = None,
    ) -> None:
        self._db = db
        self._template_engine = template_engine
        self._unit_dir = unit_dir or _SYSTEMD_UNIT_DIR
        self._daemon_reload_needed = False

    async def ensure_unit_file(self, service: Service) -> None:
        """Install the unit file for *service* if it does not already exist.

        After installing, marks that a ``systemctl --user daemon-reload``
        is needed (batched until :meth:`daemon_reload_if_needed`).
        """
        if not service.unit_name:
            return

        unit_path = self._unit_dir / service.unit_name
        if await asyncio.to_thread(unit_path.exists):
            return

        logger.info(
            "Unit file %s missing, auto-installing for service %r",
            unit_path,
            service.id,
        )

        settings = await _build_settings(self._db)
        unit_vars: dict[str, Any] | None = service.extra_config.get("unit_vars")

        try:
            rendered = self._template_engine.render_service_unit(
                service,
                settings,
                unit_vars=unit_vars,
            )
        except Exception:
            logger.exception(
                "Failed to render unit file for service %r",
                service.id,
            )
            return

        await asyncio.to_thread(self._unit_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(unit_path.write_text, rendered)
        self._daemon_reload_needed = True
        logger.info("Auto-installed unit file: %s", unit_path)

    async def daemon_reload_if_needed(self) -> None:
        """Run ``systemctl --user daemon-reload`` if any units were installed."""
        if not self._daemon_reload_needed:
            return

        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "daemon-reload",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        self._daemon_reload_needed = False
        logger.info("Ran systemctl --user daemon-reload")
