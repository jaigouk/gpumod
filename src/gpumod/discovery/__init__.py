"""Model discovery module for gpumod.

Provides automated model discovery from HuggingFace, GGUF metadata fetching,
system info collection, and preset generation for llama.cpp services.

Supports both GGUF (llama.cpp) and Safetensors (vLLM) model formats.
"""

from gpumod.discovery.gguf_metadata import GGUFFile, GGUFMetadataFetcher
from gpumod.discovery.hf_searcher import (
    HuggingFaceSearcher,
    detect_model_format,
    get_driver_hint,
)
from gpumod.discovery.llamacpp_options import LlamaCppOptions, RecommendedConfig
from gpumod.discovery.preset_generator import PresetGenerator
from gpumod.discovery.protocols import ModelSearcher, SearchResult
from gpumod.discovery.system_info import SystemInfo, SystemInfoCollector
from gpumod.discovery.unsloth_lister import HFModel, UnslothModel, UnslothModelLister

__all__ = [
    # Protocols (ISP/DIP)
    "ModelSearcher",
    "SearchResult",
    # Searchers
    "HuggingFaceSearcher",
    "UnslothModelLister",
    # Format detection
    "detect_model_format",
    "get_driver_hint",
    # Metadata
    "GGUFFile",
    "GGUFMetadataFetcher",
    "HFModel",
    "UnslothModel",
    # System info
    "SystemInfo",
    "SystemInfoCollector",
    # Preset generation
    "LlamaCppOptions",
    "PresetGenerator",
    "RecommendedConfig",
]
