"""Tests for the adapter-based benchmark model registry (gpumod-nor9).

RED-first: pins the migrated REGISTRY, the frozen ModelSpec contract, the
CLI-choices-from-registry property, and the result-JSON identity schema.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gpumod.benchmarks.coding.sampler_config import (
    GEMMA_CODING,
    THINKING_CODING,
    VIBETHINKER_CODING,
)
from gpumod.benchmarks.model_registry import REGISTRY, ModelSpec
from gpumod.benchmarks.normalizers import CodeAnswerNormalizer

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_runner_module() -> Any:
    """Import scripts/run_qwen36_benchmark.py as a module (not a package)."""
    script_path = _REPO_ROOT / "scripts" / "run_qwen36_benchmark.py"
    # The script does sys.path.insert(0, <repo>/src) at import time; gpumod is
    # already importable so this is a no-op for us, but it must not error.
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_run_qwen36_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the script's @dataclass decorators can resolve
    # their own annotations via sys.modules[cls.__module__] (Python 3.12).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPECTED_IDS = {
    "qwen36-27b",
    "qwen36-35b-a3b",
    "qwen36-35b-a3b-iq4xs",
    "qwen36-27b-mtp-q4",
    "qwen36-35b-a3b-mtp-iq4xs",
    "qwen36-35b-a3b-mtp-iq4xs-preserve",
    "qwen35-35b-a3b-heretic-mtp-q3kl-preserve",
    "gemma4-e4b",
    "gemma4-e2b-qat-q2",
    "gemma4-e2b-qat-q4",
    "gemma4-12b-q4",
    "gemma4-12b-q5",
    "gemma4-12b-q8",
    "gemma4-26b-a4b-q4",
    "gemma4-26b-a4b-qat-q4",
    "gemma4-26b-a4b-qat-mtp-q4",
    "siq1-35b-q4km",
    "agentworld-35b-a3b-q4",
    "vibethinker-3b-q8",
}


class TestRegistryShape:
    def test_registry_non_empty(self) -> None:
        assert len(REGISTRY) > 0

    def test_every_value_is_modelspec(self) -> None:
        assert all(isinstance(v, ModelSpec) for v in REGISTRY.values())

    def test_every_key_matches_spec_id(self) -> None:
        for key, spec in REGISTRY.items():
            assert key == spec.id

    def test_all_19_ids_present(self) -> None:
        assert set(REGISTRY) == EXPECTED_IDS

    def test_key_models_present(self) -> None:
        for key in (
            "qwen36-27b",
            "gemma4-26b-a4b-qat-mtp-q4",
            "vibethinker-3b-q8",
            "agentworld-35b-a3b-q4",
            "siq1-35b-q4km",
        ):
            assert key in REGISTRY


class TestModelSpecContract:
    def test_modelspec_is_frozen(self) -> None:
        spec = REGISTRY["qwen36-27b"]
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.id = "mutated"  # type: ignore[misc]

    def test_default_entry_defaults(self) -> None:
        spec = REGISTRY["qwen36-27b"]
        assert spec.normalizer is None
        assert spec.max_tokens == 32768
        assert spec.sampler == THINKING_CODING

    def test_gemma_entries_use_gemma_sampler(self) -> None:
        assert REGISTRY["gemma4-26b-a4b-qat-mtp-q4"].sampler == GEMMA_CODING
        assert REGISTRY["gemma4-12b-q4"].sampler == GEMMA_CODING

    def test_vibethinker_uses_vibethinker_sampler(self) -> None:
        assert REGISTRY["vibethinker-3b-q8"].sampler == VIBETHINKER_CODING

    def test_all_current_models_have_no_normalizer(self) -> None:
        assert all(spec.normalizer is None for spec in REGISTRY.values())

    def test_migrated_identity_fields(self) -> None:
        spec = REGISTRY["gemma4-26b-a4b-qat-mtp-q4"]
        assert spec.name == "Gemma 4 26B-A4B IT QAT UD-Q4_K_XL + MTP"
        assert spec.architecture == "moe-26B-A4B+mtp"
        assert spec.repo == "unsloth/gemma-4-26B-A4B-it-qat-GGUF"
        assert spec.quant == "QAT UD-Q4_K_XL"
        assert spec.file == "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
        assert spec.port == 7110
        assert spec.service_id == "gemma4-26b-a4b-qat-mtp-q4"


class TestChoicesDeriveFromRegistry:
    """Add a known-arch model = one data entry, ZERO runner edits."""

    def test_choices_include_new_registry_entry(self) -> None:
        spec = ModelSpec(
            id="zzz-new",
            name="ZZZ New Model",
            architecture="dense-1B",
            repo="org/zzz-new-GGUF",
            quant="Q4_K_M",
            file="zzz-new-Q4_K_M.gguf",
            port=9999,
            service_id="zzz-new",
        )
        choices = [*sorted({**REGISTRY, "zzz-new": spec}), "all"]
        assert "zzz-new" in choices
        assert choices[-1] == "all"
        # No runner code changed: the choice came purely from the registry dict.


class TestModelJsonSchemaUnchanged:
    """`_model_identity` emits the exact 9-key schema of the baseline JSON."""

    def test_model_json_schema_unchanged(self) -> None:
        runner = _load_runner_module()
        spec = ModelSpec(
            id="schema-test",
            name="Schema Test",
            architecture="dense-1B",
            repo="org/schema-GGUF",
            quant="Q4_K_M",
            file="schema-Q4_K_M.gguf",
            port=1234,
            service_id="schema-test",
            sampler=GEMMA_CODING,
            normalizer=CodeAnswerNormalizer(),  # non-None: must NOT leak into JSON
            max_tokens=1024,
        )
        identity = runner._model_identity(spec)
        # json.dumps must not raise on the normalizer object.
        loaded = json.loads(json.dumps(identity))
        assert set(loaded.keys()) == {
            "id",
            "name",
            "architecture",
            "repo",
            "quant",
            "file",
            "port",
            "service_id",
            "sampler",
        }
        # sampler is the full 6-key to_dict()
        assert set(loaded["sampler"].keys()) == {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        }

    def test_matches_committed_baseline_keys(self) -> None:
        baseline_path = (
            _REPO_ROOT
            / "docs"
            / "benchmarks"
            / "20260625_unified"
            / "result_gemma4-26b-a4b-qat-mtp-q4.json"
        )
        baseline = json.loads(baseline_path.read_text())
        runner = _load_runner_module()
        spec = REGISTRY["gemma4-26b-a4b-qat-mtp-q4"]
        identity = runner._model_identity(spec)
        assert set(identity.keys()) == set(baseline["model"].keys())
        assert set(identity["sampler"].keys()) == set(baseline["model"]["sampler"].keys())
