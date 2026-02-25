"""llama.cpp architecture compatibility checker.

Validates model architecture support against installed llama.cpp version.
Fail-open design: unknown architectures and version detection failures
result in warnings, not errors.

Part of gpumod-h1c.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Regex to parse llama-server version output: "version: 7801 (abc123def)"
VERSION_PATTERN = re.compile(r"version:\s*(\d+)")

# Mapping: HuggingFace model class -> llama.cpp architecture
# Used to translate config.json 'architectures' field to llama.cpp arch strings
HF_TO_LLAMACPP_ARCH: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen2MoeForCausalLM": "qwen2moe",
    "Qwen3MoeForCausalLM": "qwen35moe",  # Qwen 3.5 MoE uses this class
    "DeepseekV2ForCausalLM": "deepseek2",
    "Phi3ForCausalLM": "phi3",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
    "Starcoder2ForCausalLM": "starcoder2",
    "CohereForCausalLM": "command-r",
}


def get_arch_from_hf_class(hf_class: str) -> str | None:
    """Get llama.cpp architecture string from HuggingFace model class.

    Args:
        hf_class: HuggingFace model class name (e.g., "LlamaForCausalLM").

    Returns:
        llama.cpp architecture string or None if unknown.
    """
    return HF_TO_LLAMACPP_ARCH.get(hf_class)


@dataclass
class ArchSupport:
    """Architecture support entry."""

    min_version: int
    notes: str | None = None


class CompatibilityChecker:
    """Checks model architecture support against llama.cpp version.

    Fail-open design: unknown architectures and version detection failures
    result in warnings, not errors.
    """

    def __init__(self, matrix_path: Path | None = None) -> None:
        self._cached_version: int | None = None
        self._version_checked: bool = False
        # For testing - allows mocking subprocess output
        self._mock_version_output: str | None = None
        # Load architecture support matrix
        self._matrix: Mapping[str, ArchSupport] = self._load_matrix(matrix_path)

    def _load_matrix(self, path: Path | None) -> Mapping[str, ArchSupport]:
        """Load architecture support matrix from YAML."""
        if path is None:
            path = Path(__file__).parent / "data" / "arch_support.yaml"

        try:
            with path.open() as f:
                data = yaml.safe_load(f)

            matrix: dict[str, ArchSupport] = {}
            for arch, info in data.get("architectures", {}).items():
                matrix[arch] = ArchSupport(
                    min_version=info["min_version"],
                    notes=info.get("notes"),
                )
            return matrix
        except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
            logger.warning(f"Failed to load architecture matrix: {e}")
            return {}

    async def get_llamacpp_version(self) -> int | None:
        """Parse version from llama-server --version output.

        Returns:
            Build number (e.g., 7801) or None if unavailable/unparseable.
            Caches result after first call.
        """
        if self._version_checked:
            return self._cached_version

        self._version_checked = True

        # Get version output (mocked in tests, real subprocess in production)
        output = await self._get_version_output()

        if output is None:
            logger.warning("llama-server not available - version detection failed")
            return None

        # Parse version number
        match = VERSION_PATTERN.search(output)
        if match:
            self._cached_version = int(match.group(1))
            return self._cached_version

        logger.warning(f"Could not parse llama-server version from: {output}")
        return None

    async def _get_version_output(self) -> str | None:
        """Get llama-server --version output.

        Returns mock output if set (for testing), otherwise runs subprocess.
        """
        # Check for mock (testing)
        if hasattr(self, "_mock_version_output"):
            return self._mock_version_output

        # Real subprocess call (production)
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                "llama-server",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode().strip()
        except (TimeoutError, FileNotFoundError, OSError):
            return None

    async def is_supported(
        self, architecture: str, version: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if architecture is supported by llama.cpp version.

        Args:
            architecture: The model architecture string (e.g., "qwen35moe").
            version: Explicit version to check against. If None, uses cached/detected.

        Returns:
            (True, None) - supported
            (True, "warning") - unknown arch or no version info (fail-open)
            (False, "reason") - unsupported with explanation
        """
        # Get version to check against
        if version is None:
            version = await self.get_llamacpp_version()

        # Fail-open if version detection failed
        if version is None:
            return (True, "llama.cpp version unknown - assuming supported")

        # Check if architecture is in matrix
        if architecture not in self._matrix:
            return (True, f"unknown architecture '{architecture}' - assuming supported")

        # Check version requirement
        required = self._matrix[architecture].min_version
        if version >= required:
            return (True, None)

        return (
            False,
            f"{architecture} requires llama.cpp b{required}+ (you have b{version})",
        )
