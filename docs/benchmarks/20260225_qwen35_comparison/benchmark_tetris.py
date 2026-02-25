#!/usr/bin/env python3
"""Benchmark Qwen3.5 models with multi-agent Tetris development.

Compares:
- qwen35-27b-multi (Qwen3.5-27B Q4_K_XL, port 7082)
- qwen35-35b-q3-multi (Qwen3.5-35B Q3_K_XL, port 7081)
- qwen35-35b-multi (Qwen3.5-35B Q4_K_XL, port 7080)

Each model runs with 3 parallel slots.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODELS = {
    "qwen35-27b-multi": {
        "name": "Qwen3.5-27B Q4_K_XL",
        "port": 7082,
        "vram_gb": 17,
    },
    "qwen35-35b-q3-multi": {
        "name": "Qwen3.5-35B Q3_K_XL",
        "port": 7081,
        "vram_gb": 16,
    },
    "qwen35-35b-multi": {
        "name": "Qwen3.5-35B Q4_K_XL",
        "port": 7080,
        "vram_gb": 20,
    },
}


@dataclass
class PhaseResult:
    """Result from a single phase."""
    name: str
    ttft_ms: float
    total_time_ms: float
    output_chars: int
    tokens_generated: int = 0
    content_preview: str = ""


@dataclass
class TetrisResult:
    """Complete Tetris test result for a model."""
    model_id: str
    model_name: str
    port: int
    vram_gb: int
    timestamp: str
    phases: list[PhaseResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    code_valid: bool = False
    code_lines: int = 0
    qa_verdict: str = ""
    error: str | None = None


async def call_llm(
    port: int,
    role: str,
    messages: list[dict],
    max_tokens: int = 2000,
) -> tuple[str, float, float]:
    """Call LLM and return (content, ttft_ms, total_time_ms)."""
    url = f"http://localhost:{port}/v1/chat/completions"

    start = time.perf_counter()
    ttft = 0.0

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            url,
            json={
                "model": "model",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        total_time = (time.perf_counter() - start) * 1000

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Estimate TTFT from usage if available
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(content) // 4)
        if completion_tokens > 0:
            # Rough estimate: TTFT is total time minus generation time
            gen_time = total_time * 0.9  # Most time is generation
            ttft = total_time - gen_time

        print(f"  [{role}] {len(content)} chars in {total_time:.0f}ms")
        return content, ttft, total_time


async def run_tetris_test(model_id: str, config: dict) -> TetrisResult:
    """Run the full Tetris multi-agent test on a model."""
    result = TetrisResult(
        model_id=model_id,
        model_name=config["name"],
        port=config["port"],
        vram_gb=config["vram_gb"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n{'='*60}")
    print(f"Testing: {config['name']} (port {config['port']})")
    print(f"{'='*60}")

    try:
        total_start = time.perf_counter()

        # Phase 1: Planning
        print("\nPhase 1: Planning...")
        plan_messages = [{
            "role": "user",
            "content": """Design a terminal-based Tetris game in Python.

Requirements:
- Pure Python, no external dependencies (use curses for terminal)
- Standard Tetris pieces (I, O, T, S, Z, J, L)
- Arrow keys for movement, up for rotation
- Line clearing and scoring
- Game over detection

Provide a detailed architecture plan with:
1. Class structure
2. Key functions
3. Game loop design
4. Data structures for pieces and board

Be concise but complete. Output ONLY the plan, no code yet."""
        }]
        plan, plan_ttft, plan_time = await call_llm(
            config["port"], "PLANNER", plan_messages, max_tokens=1500
        )
        result.phases.append(PhaseResult(
            name="planner",
            ttft_ms=plan_ttft,
            total_time_ms=plan_time,
            output_chars=len(plan),
            content_preview=plan[:300],
        ))

        # Phase 2: Development
        print("Phase 2: Development...")
        dev_messages = [{
            "role": "user",
            "content": f"""Based on this architecture plan, implement a complete terminal Tetris game:

{plan}

Requirements:
- Single Python file
- Use curses for terminal rendering
- Must be fully functional and runnable
- Include all standard Tetris pieces
- Handle keyboard input for movement/rotation

Output ONLY the complete Python code, no explanations. Start with #!/usr/bin/env python3"""
        }]
        code, dev_ttft, dev_time = await call_llm(
            config["port"], "DEVELOPER", dev_messages, max_tokens=4000
        )

        # Extract code block if wrapped
        if "```python" in code:
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()
        elif "```" in code:
            start = code.find("```") + 3
            end = code.find("```", start)
            if end > start:
                code = code[start:end].strip()

        result.phases.append(PhaseResult(
            name="developer",
            ttft_ms=dev_ttft,
            total_time_ms=dev_time,
            output_chars=len(code),
            content_preview=code[:300],
        ))
        result.code_lines = len(code.split("\n"))

        # Save the code
        code_file = Path(__file__).parent / f"tetris_{model_id}.py"
        code_file.write_text(code)
        print(f"  Saved: {code_file.name}")

        # Validate syntax
        try:
            compile(code, "<string>", "exec")
            result.code_valid = True
            print("  Syntax: VALID")
        except SyntaxError as e:
            result.code_valid = False
            print(f"  Syntax: INVALID - {e}")

        # Phase 3: QA Review
        print("Phase 3: QA Review...")
        qa_messages = [{
            "role": "user",
            "content": f"""Review this Tetris game code for bugs and issues:

```python
{code}
```

Check for:
1. Syntax errors
2. Logic bugs (collision detection, line clearing, rotation)
3. Missing imports
4. Runtime errors
5. Playability issues

List any critical issues found. If the code looks correct, say "LGTM - code appears functional"."""
        }]
        review, qa_ttft, qa_time = await call_llm(
            config["port"], "QA", qa_messages, max_tokens=1000
        )
        result.phases.append(PhaseResult(
            name="qa",
            ttft_ms=qa_ttft,
            total_time_ms=qa_time,
            output_chars=len(review),
            content_preview=review[:300],
        ))

        # Determine QA verdict
        if "LGTM" in review or "appears functional" in review.lower():
            result.qa_verdict = "PASS"
        elif "critical" in review.lower() or "error" in review.lower():
            result.qa_verdict = "ISSUES"
        else:
            result.qa_verdict = "REVIEW"

        result.total_time_ms = (time.perf_counter() - total_start) * 1000

    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")

    return result


def print_summary(results: list[TetrisResult]) -> None:
    """Print comparison summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY: Qwen3.5 Model Comparison")
    print("="*80)

    # Header
    print(f"\n{'Model':<25} {'VRAM':>6} {'Total':>10} {'Plan':>8} {'Dev':>8} {'QA':>8} {'Lines':>6} {'Valid':>6}")
    print("-"*80)

    for r in results:
        if r.error:
            print(f"{r.model_name:<25} {'ERROR':>6} {r.error}")
            continue

        plan_time = r.phases[0].total_time_ms if len(r.phases) > 0 else 0
        dev_time = r.phases[1].total_time_ms if len(r.phases) > 1 else 0
        qa_time = r.phases[2].total_time_ms if len(r.phases) > 2 else 0

        print(
            f"{r.model_name:<25} "
            f"{r.vram_gb:>5}G "
            f"{r.total_time_ms/1000:>9.1f}s "
            f"{plan_time/1000:>7.1f}s "
            f"{dev_time/1000:>7.1f}s "
            f"{qa_time/1000:>7.1f}s "
            f"{r.code_lines:>6} "
            f"{'YES' if r.code_valid else 'NO':>6}"
        )

    print("-"*80)

    # Find winner
    valid_results = [r for r in results if not r.error and r.code_valid]
    if valid_results:
        fastest = min(valid_results, key=lambda r: r.total_time_ms)
        print(f"\nFastest (valid code): {fastest.model_name} ({fastest.total_time_ms/1000:.1f}s)")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5 models with Tetris test")
    parser.add_argument("--model", "-m", help="Test specific model ID only")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to test
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODELS.keys())}")
            return
        models_to_test = {args.model: MODELS[args.model]}
    else:
        models_to_test = MODELS

    results = []
    for model_id, config in models_to_test.items():
        result = await run_tetris_test(model_id, config)
        results.append(result)

        # Save individual result
        result_file = output_dir / f"result_{model_id}.json"
        with open(result_file, "w") as f:
            json.dump({
                "model_id": result.model_id,
                "model_name": result.model_name,
                "port": result.port,
                "vram_gb": result.vram_gb,
                "timestamp": result.timestamp,
                "total_time_ms": result.total_time_ms,
                "code_valid": result.code_valid,
                "code_lines": result.code_lines,
                "qa_verdict": result.qa_verdict,
                "phases": [
                    {
                        "name": p.name,
                        "ttft_ms": p.ttft_ms,
                        "total_time_ms": p.total_time_ms,
                        "output_chars": p.output_chars,
                    }
                    for p in result.phases
                ],
                "error": result.error,
            }, f, indent=2)
        print(f"  Result saved: {result_file.name}")

    print_summary(results)

    # Save combined results
    combined_file = output_dir / "results_combined.json"
    with open(combined_file, "w") as f:
        json.dump({
            "benchmark": "Qwen3.5 Tetris Multi-Agent Comparison",
            "date": datetime.now(timezone.utc).isoformat(),
            "hardware": "RTX 4090 (24GB)",
            "results": [
                {
                    "model_id": r.model_id,
                    "model_name": r.model_name,
                    "vram_gb": r.vram_gb,
                    "total_time_ms": r.total_time_ms,
                    "code_valid": r.code_valid,
                    "code_lines": r.code_lines,
                    "qa_verdict": r.qa_verdict,
                    "error": r.error,
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"\nCombined results: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
