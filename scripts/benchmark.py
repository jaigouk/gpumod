#!/usr/bin/env python3
"""Benchmark runner for llama.cpp router models.

Collects responses, performance metrics, and runs automated
hallucination detection on factual prompts.

Usage:
    uv run python scripts/benchmark.py --model nemotron-3-nano \
        --output docs/benchmarks/20260207_nemotron_devstral/
    uv run python scripts/benchmark.py --lifecycle \
        --models nemotron-3-nano,devstral-small-2 \
        --output docs/benchmarks/20260207_nemotron_devstral/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROUTER_URL = "http://localhost:7070"
TEMPERATURE = 0.2
TIMEOUT_S = 120
MODEL_LOAD_TIMEOUT_S = 90
VRAM_POLL_INTERVAL_S = 0.5

# Simple model name mapping (router id → short name for filenames)
MODEL_SHORT_NAMES: dict[str, str] = {
    "nemotron-3-nano": "nemotron",
    "devstral-small-2": "devstral",
    "glm-4-flash": "glm",
}


def short_name(model_id: str) -> str:
    return MODEL_SHORT_NAMES.get(model_id, model_id.split("-")[0])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VerifiableFact:
    """A single verifiable fact with keywords."""

    description: str
    required: list[str]  # ALL must appear (case-insensitive substring)
    forbidden: list[str] = field(default_factory=list)  # ANY = hallucination


@dataclass
class BenchmarkPrompt:
    """A benchmark prompt with optional ground truth for hallucination check."""

    category: str
    name: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = 1024
    ground_truth: list[VerifiableFact] | None = None


@dataclass
class HallucinationResult:
    facts_checked: int
    facts_correct: int
    facts_missing: int
    facts_hallucinated: int
    details: list[dict[str, str]]

    @property
    def hallucination_rate(self) -> float:
        denom = self.facts_correct + self.facts_hallucinated
        return self.facts_hallucinated / denom if denom > 0 else 0.0


@dataclass
class PromptResult:
    category: str
    name: str
    prompt: str
    response: str
    ttft_ms: float
    total_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    gen_tokens_per_sec: float
    prompt_tokens_per_sec: float
    vram_peak_mb: float
    hallucination: dict | None = None
    quality_score: float | None = None  # filled by evaluator


@dataclass
class LifecycleStep:
    transition: str  # e.g. "blank → nemotron-3-nano"
    unload_ms: float
    load_ms: float
    first_request_ttft_ms: float
    first_request_total_ms: float
    total_switch_ms: float
    vram_after_mb: float


# ---------------------------------------------------------------------------
# Prompt suite
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful, accurate assistant. Answer concisely."

PROMPTS: list[BenchmarkPrompt] = [
    # ── Factual Knowledge (with hallucination detection) ──────────────
    BenchmarkPrompt(
        category="factual",
        name="thermodynamics_laws",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "Explain the laws of thermodynamics. For each law, state its "
            "number (zeroth through third) and the core principle in one sentence."
        ),
        max_tokens=512,
        ground_truth=[
            VerifiableFact(
                "Zeroth law: thermal equilibrium is transitive",
                required=["zeroth", "equilibrium"],
                forbidden=[],
            ),
            VerifiableFact(
                "First law: energy conservation",
                required=["first", "energy"],
                forbidden=[],
            ),
            VerifiableFact(
                "Second law: entropy increases",
                required=["second", "entropy"],
                forbidden=[],
            ),
            VerifiableFact(
                "Third law: absolute zero",
                required=["third", "absolute zero"],
                forbidden=[],
            ),
            VerifiableFact(
                "No fabricated laws",
                required=[],
                forbidden=["fourth law", "fifth law"],
            ),
        ],
    ),
    BenchmarkPrompt(
        category="factual",
        name="solar_system_planets",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "List the 8 planets of our solar system in order from the Sun. "
            "Which is the largest and which is the smallest?"
        ),
        max_tokens=384,
        ground_truth=[
            VerifiableFact("Mercury is listed", required=["mercury"], forbidden=[]),
            VerifiableFact("Venus is listed", required=["venus"], forbidden=[]),
            VerifiableFact("Earth is listed", required=["earth"], forbidden=[]),
            VerifiableFact("Mars is listed", required=["mars"], forbidden=[]),
            VerifiableFact("Jupiter is listed", required=["jupiter"], forbidden=[]),
            VerifiableFact("Saturn is listed", required=["saturn"], forbidden=[]),
            VerifiableFact("Uranus is listed", required=["uranus"], forbidden=[]),
            VerifiableFact("Neptune is listed", required=["neptune"], forbidden=[]),
            VerifiableFact(
                "Largest is Jupiter",
                required=["jupiter"],
                forbidden=[],
            ),
            VerifiableFact(
                "Smallest is Mercury",
                required=["mercury"],
                forbidden=[],
            ),
            VerifiableFact(
                "Pluto NOT listed as planet",
                required=[],
                forbidden=["pluto"],
            ),
        ],
    ),
    BenchmarkPrompt(
        category="factual",
        name="programming_language_origins",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "When were Python, Java, and Rust first publicly released? "
            "Who created each language? Give the year and creator for each."
        ),
        max_tokens=384,
        ground_truth=[
            VerifiableFact(
                "Python creator: Guido van Rossum",
                required=["guido"],
                forbidden=[],
            ),
            VerifiableFact(
                "Python year: 1991",
                required=["1991"],
                forbidden=[],
            ),
            VerifiableFact(
                "Java creator: James Gosling",
                required=["gosling"],
                forbidden=[],
            ),
            VerifiableFact(
                "Java year: 1995",
                required=["1995"],
                forbidden=[],
            ),
            VerifiableFact(
                "Rust creator: Graydon Hoare",
                required=["hoare"],
                forbidden=[],
            ),
            VerifiableFact(
                "Rust year: 2010 or 2015",
                required=[],  # either year is acceptable
                forbidden=[],
            ),
        ],
    ),
    # ── Reasoning & Logic ─────────────────────────────────────────────
    BenchmarkPrompt(
        category="reasoning",
        name="syllogism",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "All roses are flowers. Some flowers fade quickly. "
            "Can we logically conclude that some roses fade quickly? "
            "Explain your reasoning step by step."
        ),
        max_tokens=512,
    ),
    BenchmarkPrompt(
        category="reasoning",
        name="math_word_problem",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "A train leaves station A at 9:00 AM traveling at 80 km/h. "
            "Another train leaves station B (240 km away) at 9:30 AM "
            "traveling toward A at 100 km/h. At what time do they meet? "
            "Show your work."
        ),
        max_tokens=512,
    ),
    BenchmarkPrompt(
        category="reasoning",
        name="constraint_puzzle",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "Five houses in a row are painted different colors: red, blue, "
            "green, yellow, white. The red house is immediately to the left "
            "of the blue house. The green house is somewhere to the right of "
            "the white house. The yellow house is not next to the green house. "
            "The white house is at one of the ends. "
            "What is the order of colors from left to right?"
        ),
        max_tokens=512,
    ),
    # ── Code Generation ───────────────────────────────────────────────
    BenchmarkPrompt(
        category="code",
        name="fizzbuzz_variant",
        system_prompt="You are a senior Python developer. Write clean, typed code.",
        user_prompt=(
            "Write a Python function `fizzbuzz(n: int) -> list[str]` that "
            "returns a list of strings for numbers 1 to n. For multiples of "
            "3 return 'Fizz', multiples of 5 return 'Buzz', multiples of "
            "both return 'FizzBuzz', otherwise the number as string. "
            "Include type hints and a docstring."
        ),
        max_tokens=512,
    ),
    BenchmarkPrompt(
        category="code",
        name="binary_search",
        system_prompt="You are a senior Python developer. Write clean, typed code.",
        user_prompt=(
            "Implement a generic binary search function in Python: "
            "`def binary_search(arr: list[int], target: int) -> int` "
            "that returns the index of target, or -1 if not found. "
            "Handle edge cases. Include type hints and docstring."
        ),
        max_tokens=512,
    ),
    BenchmarkPrompt(
        category="code",
        name="bug_detection",
        system_prompt="You are a code reviewer. Find and fix the bug.",
        user_prompt=(
            "Find the bug in this Python code and explain the fix:\n\n"
            "```python\n"
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    return result\n"
            "```"
        ),
        max_tokens=512,
    ),
    # ── Tool Use / Structured Output ──────────────────────────────────
    BenchmarkPrompt(
        category="tool_use",
        name="json_extraction",
        system_prompt=(
            "You are a data extraction assistant. "
            "Always respond with valid JSON only, no markdown."
        ),
        user_prompt=(
            "Extract structured data from this text:\n\n"
            '"John Smith, age 34, works at Acme Corp as a Senior Engineer '
            "since March 2021. His email is john.smith@acme.com and he "
            'manages a team of 5 people."\n\n'
            "Return JSON with keys: name, age, company, title, "
            "start_date, email, team_size"
        ),
        max_tokens=256,
    ),
    BenchmarkPrompt(
        category="tool_use",
        name="function_calling",
        system_prompt=(
            "You have access to this function:\n"
            '{"name": "get_weather", "parameters": {"location": "string", '
            '"unit": "celsius|fahrenheit"}}\n'
            "When the user asks about weather, respond with a JSON function "
            "call: {\"function\": \"get_weather\", \"arguments\": {...}}"
        ),
        user_prompt="What's the weather like in Tokyo?",
        max_tokens=128,
    ),
    # ── Writing & Summarization ───────────────────────────────────────
    BenchmarkPrompt(
        category="writing",
        name="summarization",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "Summarize the following in exactly 3 bullet points:\n\n"
            "The transformer architecture, introduced in the 2017 paper "
            "'Attention Is All You Need' by Vaswani et al., revolutionized "
            "natural language processing. It replaced recurrent neural "
            "networks with self-attention mechanisms, allowing parallel "
            "processing of sequences. The key innovation was the multi-head "
            "attention mechanism that lets the model focus on different "
            "parts of the input simultaneously. Transformers enabled models "
            "like BERT, GPT, and T5 that achieved state-of-the-art results "
            "across NLP tasks. The architecture's scalability led to the "
            "development of large language models with billions of parameters."
        ),
        max_tokens=256,
    ),
    BenchmarkPrompt(
        category="writing",
        name="technical_explanation",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "Explain what a Docker container is to someone who has never "
            "used one. Use an analogy. Keep it under 150 words."
        ),
        max_tokens=256,
    ),
]


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


def check_hallucinations(
    response: str, facts: list[VerifiableFact]
) -> HallucinationResult:
    """Check a response against ground-truth facts."""
    response_lower = response.lower()
    details: list[dict[str, str]] = []
    correct = missing = hallucinated = 0

    for fact in facts:
        has_required = not fact.required or all(
            kw.lower() in response_lower for kw in fact.required
        )
        has_forbidden = any(
            fk.lower() in response_lower for fk in fact.forbidden
        )

        if has_forbidden:
            hallucinated += 1
            status = "hallucinated"
        elif fact.required and not has_required:
            missing += 1
            status = "missing"
        else:
            correct += 1
            status = "correct"

        details.append({"fact": fact.description, "status": status})

    return HallucinationResult(
        facts_checked=len(facts),
        facts_correct=correct,
        facts_missing=missing,
        facts_hallucinated=hallucinated,
        details=details,
    )


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------


def get_vram_mb() -> float:
    """Get current GPU VRAM usage in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


async def poll_vram_peak(stop_event: asyncio.Event) -> float:
    """Poll VRAM in background, return peak usage."""
    peak = 0.0
    while not stop_event.is_set():
        current = await asyncio.to_thread(get_vram_mb)
        peak = max(peak, current)
        try:
            await asyncio.wait_for(stop_event.wait(), VRAM_POLL_INTERVAL_S)
        except asyncio.TimeoutError:
            pass
    return peak


# ---------------------------------------------------------------------------
# Router interaction
# ---------------------------------------------------------------------------


async def router_get_models(client: httpx.AsyncClient) -> list[dict]:
    """List models from the router."""
    resp = await client.get(f"{ROUTER_URL}/v1/models")
    resp.raise_for_status()
    return resp.json()["data"]


async def router_load_model(client: httpx.AsyncClient, model_id: str) -> float:
    """Load a model, return FULL load time (POST + wait for ready) in ms."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{ROUTER_URL}/models/load",
        json={"model": model_id},
        timeout=MODEL_LOAD_TIMEOUT_S,
    )
    # 400 "already loaded" is fine — treat as instant load
    if resp.status_code == 400:
        body = resp.json()
        if "already loaded" in body.get("error", {}).get("message", ""):
            return (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    # Wait until the model actually reports as loaded (async loading)
    await wait_model_loaded(client, model_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms


async def router_unload_model(client: httpx.AsyncClient, model_id: str) -> float:
    """Unload a model, return unload time in ms."""
    t0 = time.perf_counter()
    resp = await client.post(
        f"{ROUTER_URL}/models/unload",
        json={"model": model_id},
        timeout=30,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # 400 "not loaded" is fine — already unloaded
    if resp.status_code == 400:
        return elapsed_ms
    resp.raise_for_status()
    return elapsed_ms


async def wait_model_loaded(
    client: httpx.AsyncClient, model_id: str, timeout_s: float = 90
) -> None:
    """Wait until the model status is 'loaded'."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        models = await router_get_models(client)
        for m in models:
            if m["id"] == model_id and m.get("status", {}).get("value") == "loaded":
                return
        await asyncio.sleep(1.0)
    msg = f"Model {model_id} did not reach 'loaded' within {timeout_s}s"
    raise TimeoutError(msg)


async def unload_all(client: httpx.AsyncClient) -> None:
    """Unload all currently loaded models."""
    models = await router_get_models(client)
    for m in models:
        if m.get("status", {}).get("value") == "loaded":
            await router_unload_model(client, m["id"])
            await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Single prompt execution
# ---------------------------------------------------------------------------


async def run_prompt(
    client: httpx.AsyncClient,
    model_id: str,
    prompt: BenchmarkPrompt,
) -> PromptResult:
    """Run a single prompt with streaming, measure metrics."""
    print(f"  [{prompt.category}] {prompt.name} ... ", end="", flush=True)

    # Start VRAM polling
    vram_stop = asyncio.Event()
    vram_task = asyncio.create_task(poll_vram_peak(vram_stop))

    chunks: list[str] = []
    ttft_ms = 0.0
    t_start = time.perf_counter()
    first_token_seen = False
    server_timings: dict = {}

    async with client.stream(
        "POST",
        f"{ROUTER_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": prompt.max_tokens,
            "stream": True,
        },
        timeout=TIMEOUT_S,
    ) as stream:
        async for line in stream.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Capture TTFT
            delta = data.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content and not first_token_seen:
                ttft_ms = (time.perf_counter() - t_start) * 1000
                first_token_seen = True
            if content:
                chunks.append(content)

            # Capture server timings from the final chunk
            if "timings" in data:
                server_timings = data["timings"]
            # Also check usage in the last chunk
            if "usage" in data:
                server_timings["prompt_n"] = data["usage"].get(
                    "prompt_tokens", server_timings.get("prompt_n", 0)
                )
                server_timings["predicted_n"] = data["usage"].get(
                    "completion_tokens", server_timings.get("predicted_n", 0)
                )

    total_ms = (time.perf_counter() - t_start) * 1000

    # Stop VRAM polling
    vram_stop.set()
    vram_peak = await vram_task

    response_text = "".join(chunks)
    prompt_tokens = server_timings.get("prompt_n", 0)
    completion_tokens = server_timings.get("predicted_n", len(chunks))

    # Compute tok/s from server timings or fallback to client-side
    prompt_ms = server_timings.get("prompt_ms", 0)
    predicted_ms = server_timings.get("predicted_ms", 0)

    gen_tok_s = (
        (completion_tokens / (predicted_ms / 1000)) if predicted_ms > 0
        else (completion_tokens / (total_ms / 1000)) if total_ms > 0
        else 0.0
    )
    prompt_tok_s = (
        (prompt_tokens / (prompt_ms / 1000)) if prompt_ms > 0
        else 0.0
    )

    # Hallucination check for factual prompts
    hallucination_dict = None
    if prompt.ground_truth:
        h = check_hallucinations(response_text, prompt.ground_truth)
        hallucination_dict = {
            "facts_checked": h.facts_checked,
            "facts_correct": h.facts_correct,
            "facts_missing": h.facts_missing,
            "facts_hallucinated": h.facts_hallucinated,
            "hallucination_rate": round(h.hallucination_rate, 4),
            "details": h.details,
        }

    print(
        f"done ({total_ms:.0f}ms, TTFT {ttft_ms:.0f}ms, "
        f"{gen_tok_s:.1f} tok/s, VRAM {vram_peak:.0f}MB)"
    )

    return PromptResult(
        category=prompt.category,
        name=prompt.name,
        prompt=prompt.user_prompt,
        response=response_text,
        ttft_ms=round(ttft_ms, 2),
        total_time_ms=round(total_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        gen_tokens_per_sec=round(gen_tok_s, 2),
        prompt_tokens_per_sec=round(prompt_tok_s, 2),
        vram_peak_mb=round(vram_peak, 1),
        hallucination=hallucination_dict,
    )


# ---------------------------------------------------------------------------
# Full model benchmark
# ---------------------------------------------------------------------------


async def run_model_benchmark(model_id: str, output_dir: Path) -> Path:
    """Run all prompts for a single model, save results JSON."""
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    sname = short_name(model_id)
    output_path = output_dir / f"{today}_{sname}.json"

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        # Ensure clean state
        print(f"\n{'='*60}")
        print(f"Benchmark: {model_id}")
        print(f"Temperature: {TEMPERATURE}")
        print(f"{'='*60}")

        print("Unloading all models ...")
        await unload_all(client)
        await asyncio.sleep(2)

        print(f"Loading {model_id} ...")
        load_ms = await router_load_model(client, model_id)
        print(f"  Load time: {load_ms:.0f}ms")
        await asyncio.sleep(2)

        # Warm-up request (not recorded)
        print("Warm-up request ...")
        warmup = BenchmarkPrompt(
            category="_warmup",
            name="_warmup",
            system_prompt="Reply with OK.",
            user_prompt="Hello",
            max_tokens=8,
        )
        await run_prompt(client, model_id, warmup)

        # Run all prompts
        results: list[dict] = []
        for prompt in PROMPTS:
            result = await run_prompt(client, model_id, prompt)
            results.append(asdict(result))

        # Compute summary
        summary = _compute_summary(results, load_ms)

        output = {
            "metadata": {
                "model": model_id,
                "short_name": sname,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "temperature": TEMPERATURE,
                "router_url": ROUTER_URL,
            },
            "results": results,
            "load_ms": round(load_ms, 2),
            "summary": summary,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nResults saved: {output_path}")

        # Unload after benchmark
        await router_unload_model(client, model_id)

    return output_path


def _compute_summary(results: list[dict], load_ms: float) -> dict:
    """Compute aggregate summary from prompt results."""
    categories: dict[str, list[float]] = {}
    ttfts = []
    gen_speeds = []
    total_hallucinated = 0
    total_facts_checked = 0

    for r in results:
        cat = r["category"]
        if cat.startswith("_"):
            continue
        ttfts.append(r["ttft_ms"])
        if r["gen_tokens_per_sec"] > 0:
            gen_speeds.append(r["gen_tokens_per_sec"])

        if r.get("hallucination"):
            total_hallucinated += r["hallucination"]["facts_hallucinated"]
            total_facts_checked += r["hallucination"]["facts_checked"]

    return {
        "load_ms": round(load_ms, 2),
        "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else 0,
        "avg_gen_tokens_per_sec": (
            round(sum(gen_speeds) / len(gen_speeds), 2) if gen_speeds else 0
        ),
        "total_facts_checked": total_facts_checked,
        "total_facts_hallucinated": total_hallucinated,
        "hallucination_rate": (
            round(total_hallucinated / total_facts_checked, 4)
            if total_facts_checked > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Lifecycle test
# ---------------------------------------------------------------------------

LIFECYCLE_VERIFY_PROMPT = BenchmarkPrompt(
    category="_lifecycle",
    name="verify",
    system_prompt="Reply with exactly: READY",
    user_prompt="Are you loaded?",
    max_tokens=16,
)


async def run_lifecycle_test(
    model_ids: list[str], output_dir: Path
) -> Path:
    """Run blank → model → RAG → model lifecycle for each model."""
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    output_path = output_dir / f"{today}_lifecycle.json"

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        print(f"\n{'='*60}")
        print("Lifecycle Test: Mode Switching")
        print(f"{'='*60}")

        all_steps: dict[str, list[dict]] = {}

        for model_id in model_ids:
            print(f"\n--- {model_id} ---")
            steps: list[LifecycleStep] = []

            # Step 1: blank → model
            print("  Ensuring blank state ...")
            await unload_all(client)
            await asyncio.sleep(2)

            print(f"  blank → {model_id} ...")
            load_ms = await router_load_model(client, model_id)
            await asyncio.sleep(1)

            verify = await run_prompt(client, model_id, LIFECYCLE_VERIFY_PROMPT)
            vram = get_vram_mb()

            steps.append(LifecycleStep(
                transition=f"blank → {model_id}",
                unload_ms=0,
                load_ms=round(load_ms, 2),
                first_request_ttft_ms=verify.ttft_ms,
                first_request_total_ms=verify.total_time_ms,
                total_switch_ms=round(load_ms + verify.total_time_ms, 2),
                vram_after_mb=vram,
            ))
            print(
                f"    Load: {load_ms:.0f}ms, "
                f"TTFT: {verify.ttft_ms:.0f}ms, "
                f"VRAM: {vram:.0f}MB"
            )

            # Step 2: model → RAG (simulate with embedding model or
            #          just unload/load cycle if no embedding available)
            print(f"  {model_id} → RAG (unload + measure gap) ...")
            unload_ms = await router_unload_model(client, model_id)
            await asyncio.sleep(1)
            vram_after_unload = get_vram_mb()

            steps.append(LifecycleStep(
                transition=f"{model_id} → RAG (unload)",
                unload_ms=round(unload_ms, 2),
                load_ms=0,
                first_request_ttft_ms=0,
                first_request_total_ms=0,
                total_switch_ms=round(unload_ms, 2),
                vram_after_mb=vram_after_unload,
            ))
            print(
                f"    Unload: {unload_ms:.0f}ms, "
                f"VRAM after: {vram_after_unload:.0f}MB"
            )

            # Step 3: RAG → model (reload)
            print(f"  RAG → {model_id} (reload) ...")
            reload_ms = await router_load_model(client, model_id)
            await asyncio.sleep(1)

            verify2 = await run_prompt(client, model_id, LIFECYCLE_VERIFY_PROMPT)
            vram2 = get_vram_mb()

            steps.append(LifecycleStep(
                transition=f"RAG → {model_id}",
                unload_ms=0,
                load_ms=round(reload_ms, 2),
                first_request_ttft_ms=verify2.ttft_ms,
                first_request_total_ms=verify2.total_time_ms,
                total_switch_ms=round(reload_ms + verify2.total_time_ms, 2),
                vram_after_mb=vram2,
            ))
            print(
                f"    Load: {reload_ms:.0f}ms, "
                f"TTFT: {verify2.ttft_ms:.0f}ms, "
                f"VRAM: {vram2:.0f}MB"
            )

            # Clean up
            await router_unload_model(client, model_id)
            all_steps[model_id] = [asdict(s) for s in steps]

        output = {
            "metadata": {
                "type": "lifecycle",
                "models": model_ids,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "router_url": ROUTER_URL,
            },
            "lifecycle": all_steps,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\nLifecycle results saved: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs on llama.cpp router"
    )
    parser.add_argument("--model", help="Model ID to benchmark")
    parser.add_argument(
        "--models",
        help="Comma-separated model IDs (for lifecycle test)",
    )
    parser.add_argument(
        "--lifecycle",
        action="store_true",
        help="Run lifecycle (mode-switching) test",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmarks/",
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.lifecycle:
        if not args.models:
            parser.error("--lifecycle requires --models")
        model_ids = [m.strip() for m in args.models.split(",")]
        asyncio.run(run_lifecycle_test(model_ids, output_dir))
    elif args.model:
        asyncio.run(run_model_benchmark(args.model, output_dir))
    else:
        parser.error("Specify --model or --lifecycle --models")


if __name__ == "__main__":
    main()
