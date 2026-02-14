#!/usr/bin/env python3
"""Multi-Agent Benchmark v2 - Improved with TTFT, reasoning model support, and auto-scoring.

Features:
- Streams responses to capture accurate TTFT
- Extracts reasoning_content for reasoning models (GPT-OSS, DeepSeek R1)
- Runs 3 bug scenarios with varying difficulty
- Auto-scores based on keyword detection
- Outputs structured JSON results

Usage:
    python multi_agent_benchmark.py                    # Uses localhost:7070
    python multi_agent_benchmark.py --port 8080       # Custom port
    python multi_agent_benchmark.py --scenario easy   # Single scenario
    python multi_agent_benchmark.py --model-name gpt-oss-20b  # Tag results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import httpx

# Agent personas
AGENTS = {
    "architect": "You are a software architect. Focus on design patterns, modularity, and potential structural issues. Be concise but thorough.",
    "developer": "You are a senior developer. Focus on implementation details, code correctness, and runtime behavior. Be concise but thorough.",
    "tester": "You are a QA engineer. Focus on edge cases, error handling, and how the code could fail. Be concise but thorough.",
    "security": "You are a security engineer. Focus on vulnerabilities, input validation, and safe practices. Be concise but thorough.",
    "reviewer": "You are a code reviewer. Synthesize the analysis and identify the most critical bug. Be concise but thorough.",
}

# Bug scenarios
SCENARIOS = {
    "easy": {
        "module": "bugs.easy_pagination",
        "name": "Off-by-one Pagination",
        "difficulty": "easy",
    },
    "medium": {
        "module": "bugs.medium_resource_leak",
        "name": "Resource Leak",
        "difficulty": "medium",
    },
    "hard": {
        "module": "bugs.hard_async_state",
        "name": "Async State Corruption",
        "difficulty": "hard",
    },
}


@dataclass
class AgentResult:
    """Result from a single agent's analysis."""

    role: str
    ttft_ms: float | None = None
    response_time_ms: float = 0
    tokens_generated: int = 0
    tokens_per_sec: float = 0
    content: str = ""
    reasoning_content: str = ""
    score: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Result from a complete benchmark run."""

    model_name: str
    scenario: str
    difficulty: str
    timestamp: str
    total_time_ms: float
    agents: list[AgentResult] = field(default_factory=list)
    aggregate_score: dict = field(default_factory=dict)
    buggy_code_fails: bool = False
    fixed_code_passes: bool = False


async def stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> tuple[str, str, float | None, float, int]:
    """Stream a completion and measure TTFT.

    Returns:
        (content, reasoning_content, ttft_ms, total_time_ms, token_count)
    """
    start_time = time.perf_counter()
    ttft_ms: float | None = None
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    token_count = 0

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,
    }

    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120.0,
    ) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Capture TTFT on first content
                if ttft_ms is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft_ms = (time.perf_counter() - start_time) * 1000

                if delta.get("content"):
                    content_chunks.append(delta["content"])
                    token_count += 1

                if delta.get("reasoning_content"):
                    reasoning_chunks.append(delta["reasoning_content"])
                    token_count += 1

            except json.JSONDecodeError:
                continue

    total_time_ms = (time.perf_counter() - start_time) * 1000
    content = "".join(content_chunks)
    reasoning = "".join(reasoning_chunks)

    return content, reasoning, ttft_ms, total_time_ms, token_count


async def run_agent(
    client: httpx.AsyncClient,
    base_url: str,
    role: str,
    system_prompt: str,
    buggy_code: str,
    score_fn: Any,
) -> AgentResult:
    """Run a single agent's analysis."""
    user_prompt = f"""Analyze this code and identify any bugs. What is wrong and how would you fix it?

```python
{buggy_code}
```

Be specific about:
1. What the bug is
2. Why it's a problem
3. How to fix it"""

    try:
        content, reasoning, ttft_ms, total_ms, tokens = await stream_completion(
            client, base_url, system_prompt, user_prompt
        )

        # Combine content for scoring (reasoning models put analysis in reasoning_content)
        full_response = content + " " + reasoning
        score = score_fn(full_response)

        return AgentResult(
            role=role,
            ttft_ms=ttft_ms,
            response_time_ms=total_ms,
            tokens_generated=tokens,
            tokens_per_sec=tokens / (total_ms / 1000) if total_ms > 0 else 0,
            content=content,
            reasoning_content=reasoning,
            score=score,
        )

    except Exception as e:
        return AgentResult(role=role, error=str(e))


async def run_scenario(
    base_url: str,
    scenario_key: str,
    model_name: str,
) -> BenchmarkResult:
    """Run all agents on a single scenario."""
    scenario = SCENARIOS[scenario_key]

    # Import the bug module
    module = import_module(scenario["module"])
    buggy_code = module.BUGGY_CODE
    score_fn = module.score_response

    # Test that buggy code fails and fixed code passes
    if hasattr(module, "test_pagination"):
        buggy_fails = not module.test_pagination(use_fixed=False)[0]
        fixed_passes = module.test_pagination(use_fixed=True)[0]
    elif hasattr(module, "test_resource_leak"):
        buggy_fails = not module.test_resource_leak(use_fixed=False)[0]
        fixed_passes = module.test_resource_leak(use_fixed=True)[0]
    elif hasattr(module, "test_sync_wrapper"):
        buggy_fails = not module.test_sync_wrapper(use_fixed=False)[0]
        fixed_passes = module.test_sync_wrapper(use_fixed=True)[0]
    else:
        buggy_fails = False
        fixed_passes = False

    # Run all agents concurrently
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            run_agent(client, base_url, role, prompt, buggy_code, score_fn)
            for role, prompt in AGENTS.items()
        ]
        agents = await asyncio.gather(*tasks)

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Aggregate scores
    total_score = sum(a.score.get("score", 0) for a in agents if a.score)
    max_score = sum(a.score.get("max_score", 3) for a in agents if a.score)
    bugs_found = sum(1 for a in agents if a.score.get("bug_identified", False))
    fixes_proposed = sum(1 for a in agents if a.score.get("fix_proposed", False))

    return BenchmarkResult(
        model_name=model_name,
        scenario=scenario["name"],
        difficulty=scenario["difficulty"],
        timestamp=datetime.now(UTC).isoformat(),
        total_time_ms=total_time_ms,
        agents=list(agents),
        aggregate_score={
            "total_score": total_score,
            "max_score": max_score,
            "bugs_found": f"{bugs_found}/5",
            "fixes_proposed": f"{fixes_proposed}/5",
            "percentage": f"{total_score / max_score * 100:.1f}%" if max_score > 0 else "N/A",
        },
        buggy_code_fails=buggy_fails,
        fixed_code_passes=fixed_passes,
    )


def print_result(result: BenchmarkResult) -> None:
    """Pretty-print a benchmark result."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULT: {result.scenario} ({result.difficulty})")
    print(f"Model: {result.model_name}")
    print(f"Time: {result.total_time_ms:.0f}ms")
    print(f"{'=' * 60}")

    print(f"\n{'Agent':<12} {'TTFT':>8} {'Time':>8} {'Tokens':>8} {'tok/s':>8} {'Score':>8}")
    print("-" * 60)

    for agent in result.agents:
        if agent.error:
            print(f"{agent.role:<12} ERROR: {agent.error}")
            continue

        ttft = f"{agent.ttft_ms:.0f}ms" if agent.ttft_ms else "-"
        time_str = f"{agent.response_time_ms:.0f}ms"
        score_str = f"{agent.score.get('score', 0)}/{agent.score.get('max_score', 3)}"

        print(
            f"{agent.role:<12} {ttft:>8} {time_str:>8} "
            f"{agent.tokens_generated:>8} {agent.tokens_per_sec:>7.0f} {score_str:>8}"
        )

    print("-" * 60)
    print(f"Aggregate: {result.aggregate_score}")
    print(
        f"Verification: buggy_fails={result.buggy_code_fails}, fixed_passes={result.fixed_code_passes}"
    )

    # Show reasoning content if present (for reasoning models)
    reasoning_agents = [a for a in result.agents if a.reasoning_content]
    if reasoning_agents:
        print(
            f"\nNote: {len(reasoning_agents)} agents returned reasoning_content (reasoning model detected)"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Agent Benchmark v2")
    parser.add_argument("--port", type=int, default=7070, help="LLM server port")
    parser.add_argument("--host", default="localhost", help="LLM server host")
    parser.add_argument("--model-name", default="unknown", help="Model name for results")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run single scenario")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Verify server is reachable
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code != 200:
                print(f"Error: Server at {base_url} returned status {resp.status_code}")
                return 1
        except httpx.ConnectError:
            print(f"Error: Cannot connect to {base_url}")
            return 1

    print(f"Connected to {base_url}")
    print(f"Model: {args.model_name}")

    # Run scenarios
    scenarios_to_run = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    results: list[BenchmarkResult] = []

    for scenario_key in scenarios_to_run:
        print(f"\nRunning scenario: {scenario_key}...")
        result = await run_scenario(base_url, scenario_key, args.model_name)
        results.append(result)
        print_result(result)

    # Save results
    if args.output:
        output_data = [asdict(r) for r in results]
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<25} {'Bugs Found':<12} {'Score':<10} {'Time':<10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r.scenario:<25} {r.aggregate_score['bugs_found']:<12} "
            f"{r.aggregate_score['percentage']:<10} {r.total_time_ms:.0f}ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
