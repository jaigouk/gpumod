#!/usr/bin/env python3
"""Mixed workload benchmark: primary coding task + concurrent side tasks.

Measures quality degradation as concurrency increases from 1 to 5 slots.
Tests GPT-OSS 20B and Qwen3-Coder 30B on realistic daily-use scenarios.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE_URL = "http://localhost:7070/v1/chat/completions"
TIMEOUT = 120.0
MAX_TOKENS_PRIMARY = 1024
MAX_TOKENS_SIDE = 512

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PRIMARY_TURNS = [
    {
        "system": (
            "You are a senior Python developer. Write clean, correct code. "
            "Be concise — code with minimal commentary."
        ),
        "user": (
            "Create a FastAPI endpoint `POST /users` that accepts a JSON body "
            "with fields: name (str), email (str), age (int). Use a Pydantic "
            "model for validation. Return 201 with the created user including "
            "a generated UUID id. Include the complete runnable code."
        ),
    },
    {
        "user": (
            "Now add email format validation (must contain @ and a dot after @), "
            "age range validation (0-150), and name must be non-empty. "
            "Return proper 422 error responses with clear messages. "
            "Show the updated Pydantic model only."
        ),
    },
    {
        "user": (
            "Add a `GET /users/{user_id}` endpoint. Store users in a module-level "
            "dict. Return 404 with `{\"detail\": \"User not found\"}` if the ID "
            "doesn't exist. Show only the new endpoint and the storage dict."
        ),
    },
]

SIDE_TASKS = {
    "S1_docstring": {
        "system": "You are a Python documentation expert. Write clear, accurate docstrings.",
        "user": (
            "Write a Google-style docstring for this function. Return ONLY the "
            "docstring, no other code.\n\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
            "```"
        ),
        "verify_keywords": ["sorted", "index", "target", "Args", "Returns"],
    },
    "S2_unit_tests": {
        "system": "You are a QA engineer. Write pytest tests. Be concise.",
        "user": (
            "Write 3 pytest unit tests for this function:\n\n"
            "```python\n"
            "def calculate_discount(price: float, discount_pct: float) -> float:\n"
            '    if not 0 <= discount_pct <= 100:\n'
            '        raise ValueError("Discount must be 0-100")\n'
            "    return round(price * (1 - discount_pct / 100), 2)\n"
            "```\n\n"
            "Test: normal case, edge case (0% and 100%), and invalid input."
        ),
        "verify_keywords": ["def test_", "pytest", "assert"],
    },
    "S3_find_bug": {
        "system": "You are a code reviewer. Identify bugs precisely.",
        "user": (
            "Find the bug in this pagination function:\n\n"
            "```python\n"
            "def paginate(items, page_size, page_num):\n"
            '    """Return items for the given page (1-indexed)."""\n'
            "    total_pages = len(items) // page_size\n"
            "    if page_num < 1 or page_num > total_pages:\n"
            '        return []\n'
            "    start = (page_num - 1) * page_size\n"
            "    return items[start:start + page_size]\n"
            "```\n\n"
            "What is wrong and how would you fix it?"
        ),
        "verify_keywords": ["ceil", "last page", "remainder"],
    },
    "S4_async_convert": {
        "system": "You are a Python async expert. Convert code to async correctly.",
        "user": (
            "Convert this synchronous function to async using aiofiles:\n\n"
            "```python\n"
            "import json\n\n"
            "def read_config(path: str) -> dict:\n"
            "    with open(path) as f:\n"
            "        data = json.load(f)\n"
            "    return data\n"
            "```\n\n"
            "Show only the async version."
        ),
        "verify_keywords": ["async def", "await", "aiofiles"],
    },
}

CONCURRENCY_LEVELS = [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    role: str
    ttft_ms: float | None
    response_time_ms: float
    tokens_generated: int
    tokens_per_sec: float
    content: str
    reasoning_content: str = ""
    verified: bool | None = None
    verify_keywords_found: list[str] = field(default_factory=list)
    verify_keywords_missing: list[str] = field(default_factory=list)


@dataclass
class ConcurrencyResult:
    concurrency: int
    primary_results: list[RequestResult]
    side_results: list[RequestResult]
    total_time_ms: float


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def send_chat(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int,
    role_label: str,
) -> RequestResult:
    """Send a chat completion request and measure timing."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    start = time.monotonic()
    ttft = None

    # Use streaming to measure TTFT
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    try:
        async with client.stream(
            "POST",
            BASE_URL,
            json={**payload, "stream": True},
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")

                if (text or reasoning) and ttft is None:
                    ttft = (time.monotonic() - start) * 1000

                if text:
                    content_parts.append(text)
                if reasoning:
                    reasoning_parts.append(reasoning)

    except httpx.HTTPStatusError as e:
        elapsed = (time.monotonic() - start) * 1000
        return RequestResult(
            role=role_label,
            ttft_ms=None,
            response_time_ms=elapsed,
            tokens_generated=0,
            tokens_per_sec=0.0,
            content=f"ERROR: {e.response.status_code}",
        )

    elapsed = (time.monotonic() - start) * 1000
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    # Rough token estimate: split on whitespace
    tokens = len(content.split()) + len(reasoning.split())
    tok_s = tokens / (elapsed / 1000) if elapsed > 0 else 0

    return RequestResult(
        role=role_label,
        ttft_ms=round(ttft, 1) if ttft else None,
        response_time_ms=round(elapsed, 1),
        tokens_generated=tokens,
        tokens_per_sec=round(tok_s, 1),
        content=content,
        reasoning_content=reasoning,
    )


def verify_side_task(result: RequestResult, keywords: list[str]) -> RequestResult:
    """Check if required keywords appear in the response."""
    text = (result.content + " " + result.reasoning_content).lower()
    found = [k for k in keywords if k.lower() in text]
    missing = [k for k in keywords if k.lower() not in text]
    result.verified = len(missing) == 0
    result.verify_keywords_found = found
    result.verify_keywords_missing = missing
    return result


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def run_primary_task(client: httpx.AsyncClient) -> list[RequestResult]:
    """Run the multi-turn primary coding conversation."""
    results: list[RequestResult] = []
    messages: list[dict] = []

    for i, turn in enumerate(PRIMARY_TURNS):
        if "system" in turn:
            messages.append({"role": "system", "content": turn["system"]})
        messages.append({"role": "user", "content": turn["user"]})

        result = await send_chat(
            client, messages, MAX_TOKENS_PRIMARY, f"primary_turn_{i + 1}"
        )
        results.append(result)

        # Add assistant response to conversation for next turn
        messages.append({"role": "assistant", "content": result.content})

    return results


async def run_side_task(
    client: httpx.AsyncClient, task_id: str, task: dict
) -> RequestResult:
    """Run a single side task."""
    messages = []
    if "system" in task:
        messages.append({"role": "system", "content": task["system"]})
    messages.append({"role": "user", "content": task["user"]})

    result = await send_chat(client, messages, MAX_TOKENS_SIDE, task_id)

    if "verify_keywords" in task:
        verify_side_task(result, task["verify_keywords"])

    return result


async def run_concurrency_level(
    client: httpx.AsyncClient, concurrency: int
) -> ConcurrencyResult:
    """Run primary task + N-1 side tasks concurrently."""
    side_task_ids = list(SIDE_TASKS.keys())
    num_side = concurrency - 1  # primary takes 1 slot

    start = time.monotonic()

    if num_side == 0:
        # Concurrency 1: primary only
        primary_results = await run_primary_task(client)
        side_results = []
    else:
        # Run primary + side tasks concurrently
        selected_sides = side_task_ids[:num_side]
        tasks = [run_primary_task(client)]
        for sid in selected_sides:
            tasks.append(run_side_task(client, sid, SIDE_TASKS[sid]))

        results = await asyncio.gather(*tasks)
        primary_results = results[0]
        side_results = list(results[1:])

    total_ms = (time.monotonic() - start) * 1000

    return ConcurrencyResult(
        concurrency=concurrency,
        primary_results=primary_results,
        side_results=side_results,
        total_time_ms=round(total_ms, 1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def result_to_dict(r: RequestResult) -> dict:
    """Convert RequestResult to a JSON-serializable dict."""
    d = {
        "role": r.role,
        "ttft_ms": r.ttft_ms,
        "response_time_ms": r.response_time_ms,
        "tokens_generated": r.tokens_generated,
        "tokens_per_sec": r.tokens_per_sec,
    }
    if r.verified is not None:
        d["verified"] = r.verified
        d["keywords_found"] = r.verify_keywords_found
        d["keywords_missing"] = r.verify_keywords_missing
    # Store content for later scoring (truncate for JSON readability)
    d["content_preview"] = r.content[:500] if r.content else ""
    if r.reasoning_content:
        d["reasoning_preview"] = r.reasoning_content[:500]
    return d


async def run_benchmark(model_name: str, output_dir: Path) -> dict:
    """Run full benchmark for one model across all concurrency levels."""
    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {model_name}")
    print(f"{'=' * 60}\n")

    all_results = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": "RTX 4090 (24GB)",
        "concurrency_levels": [],
    }

    async with httpx.AsyncClient() as client:
        # Warm up: send a tiny request to ensure model is loaded
        print("Warming up...")
        warmup = await send_chat(
            client,
            [{"role": "user", "content": "Say hello."}],
            16,
            "warmup",
        )
        print(f"  Warmup TTFT: {warmup.ttft_ms}ms, Time: {warmup.response_time_ms}ms")

        for level in CONCURRENCY_LEVELS:
            print(f"\n--- Concurrency level {level} ---")
            result = await run_concurrency_level(client, level)

            level_data = {
                "concurrency": level,
                "total_time_ms": result.total_time_ms,
                "primary": [result_to_dict(r) for r in result.primary_results],
                "side_tasks": [result_to_dict(r) for r in result.side_results],
            }

            # Print summary
            for pr in result.primary_results:
                print(
                    f"  {pr.role}: {pr.response_time_ms}ms, "
                    f"{pr.tokens_generated} tok, {pr.tokens_per_sec} tok/s"
                )
            for sr in result.side_results:
                status = "PASS" if sr.verified else "FAIL"
                print(
                    f"  {sr.role}: {sr.response_time_ms}ms, "
                    f"{sr.tokens_generated} tok, [{status}]"
                )
            print(f"  Total: {result.total_time_ms}ms")

            all_results["concurrency_levels"].append(level_data)

            # Brief pause between levels to let the server settle
            await asyncio.sleep(2)

    # Save results
    output_file = output_dir / f"20260222_{model_name.replace(' ', '_').lower()}.json"
    output_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_file}")

    return all_results


async def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    await run_benchmark(model_name, output_dir)


if __name__ == "__main__":
    asyncio.run(main())
