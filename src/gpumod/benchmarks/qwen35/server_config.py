"""llama.cpp server configuration for Qwen3.5 models.

Based on verified 2026 recommendations:
- https://github.com/ggml-org/llama.cpp/issues/11200
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    """Immutable llama.cpp server configuration."""

    # KV cache quantization (verified: +50-100% speed, minimal quality loss)
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"

    # Performance flags
    flash_attention: bool = True
    fit: bool = True
    no_mmap: bool = True

    # Chat template
    jinja: bool = True

    # Resources
    context_size: int = 32768
    threads: int = 16

    def to_cli_args(self) -> list[str]:
        """Convert to llama.cpp CLI arguments.

        Returns:
            List of CLI argument strings
        """
        args: list[str] = []

        # KV cache quantization
        args.extend(["-ctk", self.cache_type_k])
        args.extend(["-ctv", self.cache_type_v])

        # Performance flags
        if self.flash_attention:
            args.extend(["-fa", "on"])

        if self.fit:
            args.extend(["--fit", "on"])

        if self.no_mmap:
            args.append("--no-mmap")

        # Chat template
        if self.jinja:
            args.append("--jinja")

        # Resources
        args.extend(["-c", str(self.context_size)])
        args.extend(["-t", str(self.threads)])

        return args


# Preset for 24GB VRAM cards (RTX 4090, 3090)
DEFAULT_24GB = ServerConfig(
    cache_type_k="q8_0",
    cache_type_v="q8_0",
    flash_attention=True,
    fit=True,
    no_mmap=True,
    jinja=True,
    context_size=65536,
    threads=16,
)

# Preset for 16GB VRAM cards (RTX 4080, 5080)
DEFAULT_16GB = ServerConfig(
    cache_type_k="q8_0",
    cache_type_v="q8_0",
    flash_attention=True,
    fit=True,
    no_mmap=True,
    jinja=True,
    context_size=32768,
    threads=16,
)

# Registry of available configs
_CONFIGS: dict[str, ServerConfig] = {
    "24gb": DEFAULT_24GB,
    "16gb": DEFAULT_16GB,
}


def get_server_config(name: str) -> ServerConfig:
    """Get a server config by name.

    Args:
        name: Config name ("24gb" or "16gb"), case-insensitive

    Returns:
        The corresponding ServerConfig

    Raises:
        ValueError: If name is not recognized
    """
    key = name.lower()
    if key not in _CONFIGS:
        valid = ", ".join(_CONFIGS.keys())
        msg = f"Unknown config: {name}. Valid options: {valid}"
        raise ValueError(msg)
    return _CONFIGS[key]
