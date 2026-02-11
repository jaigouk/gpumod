"""RLM consulting module for gpumod.

Provides multi-step reasoning over GPU/model data via a
Recursive Language Model (RLM) environment.
"""

from gpumod.rlm.environment import GpumodConsultEnv
from gpumod.rlm.orchestrator import ConsultResult, RLMOrchestrator
from gpumod.rlm.security import SecurityError

__all__ = [
    "ConsultResult",
    "GpumodConsultEnv",
    "RLMOrchestrator",
    "SecurityError",
]
