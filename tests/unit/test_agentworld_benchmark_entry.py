"""Guard test for the AgentWorld entry in the coding-suite runner (gpumod-qsgl.4).

``scripts/run_qwen36_benchmark.py`` is a utility script, not part of the
``gpumod`` package, so it is loaded via ``importlib`` rather than imported. The
test pins the two things that silently break a benchmark arm: the ``MODELS``
entry (with the correct sampler) and the ``--model`` CLI choice.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_qwen36_benchmark.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_qwen36_benchmark", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve annotations via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentworld_entry_present_with_thinking_sampler() -> None:
    runner = _load_runner()
    assert "agentworld-35b-a3b-q4" in runner.MODELS
    cfg = runner.MODELS["agentworld-35b-a3b-q4"]
    # Qwen reasoning MoE -> THINKING_CODING, not GEMMA_CODING.
    assert cfg.sampler is runner.THINKING_CODING
    assert cfg.service_id == "agentworld-35b-a3b-q4"
    assert cfg.port == 7111
    assert cfg.repo == "gaoqianshen/Qwen-AgentWorld-35B-A3B-Q4_K_M-GGUF"


def test_cli_accepts_agentworld_model_choice(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    runner = _load_runner()
    monkeypatch.setattr(
        sys, "argv", ["run_qwen36_benchmark.py", "--model", "agentworld-35b-a3b-q4"]
    )
    args = runner.parse_args()
    assert args.model == "agentworld-35b-a3b-q4"
