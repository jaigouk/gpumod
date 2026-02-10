"""Model discovery module for gpumod.

Provides automated model discovery from HuggingFace, GGUF metadata fetching,
system info collection, and preset generation for llama.cpp services.
"""

from gpumod.discovery.gguf_metadata import GGUFFile, GGUFMetadataFetcher
from gpumod.discovery.llamacpp_options import LlamaCppOptions, RecommendedConfig
from gpumod.discovery.preset_generator import PresetGenerator
from gpumod.discovery.system_info import SystemInfo, SystemInfoCollector
from gpumod.discovery.unsloth_lister import UnslothModel, UnslothModelLister

__all__ = [
    "GGUFFile",
    "GGUFMetadataFetcher",
    "LlamaCppOptions",
    "PresetGenerator",
    "RecommendedConfig",
    "SystemInfo",
    "SystemInfoCollector",
    "UnslothModel",
    "UnslothModelLister",
]
