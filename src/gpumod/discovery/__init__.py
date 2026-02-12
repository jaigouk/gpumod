"""Model discovery module for gpumod.

Provides automated model discovery from HuggingFace, GGUF metadata fetching,
system info collection, and preset generation for llama.cpp services.

Supports both GGUF (llama.cpp) and Safetensors (vLLM) model formats.
"""

from gpumod.discovery.config_fetcher import ConfigFetcher, ConfigNotFoundError, ModelConfig
from gpumod.discovery.content_truncator import (
    CharTruncator,
    ContentTruncator,
    TokenTruncator,
    TruncationResult,
)
from gpumod.discovery.docs_fetcher import DocsNotFoundError, DriverDocs, DriverDocsFetcher
from gpumod.discovery.gguf_metadata import GGUFFile, GGUFMetadataFetcher
from gpumod.discovery.hf_searcher import (
    HuggingFaceSearcher,
    detect_model_format,
    get_driver_hint,
)
from gpumod.discovery.llamacpp_options import LlamaCppOptions, RecommendedConfig
from gpumod.discovery.preset_generator import PresetGenerator
from gpumod.discovery.protocols import ModelSearcher, SearchResult
from gpumod.discovery.section_filter import (
    ExactMatcher,
    FuzzyMatcher,
    SectionFilter,
    SectionMatch,
    SectionNotFoundError,
)
from gpumod.discovery.system_info import SystemInfo, SystemInfoCollector
from gpumod.discovery.unsloth_lister import HFModel, UnslothModel, UnslothModelLister

__all__ = [
    # Content truncation (gpumod-v7k)
    "CharTruncator",
    # Config fetching
    "ConfigFetcher",
    "ConfigNotFoundError",
    "ContentTruncator",
    # Docs fetching
    "DocsNotFoundError",
    "DriverDocs",
    "DriverDocsFetcher",
    # Section filtering (gpumod-v7k)
    "ExactMatcher",
    "FuzzyMatcher",
    # GGUF metadata
    "GGUFFile",
    "GGUFMetadataFetcher",
    # HuggingFace search
    "HFModel",
    "HuggingFaceSearcher",
    # Llama.cpp options
    "LlamaCppOptions",
    # Model protocols
    "ModelConfig",
    "ModelSearcher",
    # Preset generation
    "PresetGenerator",
    "RecommendedConfig",
    "SearchResult",
    "SectionFilter",
    "SectionMatch",
    "SectionNotFoundError",
    # System info
    "SystemInfo",
    "SystemInfoCollector",
    "TokenTruncator",
    "TruncationResult",
    # Unsloth models
    "UnslothModel",
    "UnslothModelLister",
    "detect_model_format",
    "get_driver_hint",
]
