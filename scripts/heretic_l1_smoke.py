"""L1 flake-rate smoke check for the heretic Q3_K_L variant (gpumod-ojey).

The main benchmark in scripts/run_qwen36_benchmark.py showed the heretic
preserve_thinking run failing L1 (Basic Queue) in 4/15 iterations while the
baseline only failed 1/15. This script hammers the same L1 prompt many more
times (default 20) against the running heretic service to nail down the
true pass rate with a tighter Wilson 95% CI, so we can decide whether the
flake is benign noise or a real competence regression.

It is intentionally standalone — it does not touch the registered runner,
preset matrix, or benchmark output dirs.

Usage:
    uv run python -u scripts/heretic_l1_smoke.py \\
        --base-url http://localhost:7105 \\
        --attempts 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from gpumod.benchmarks.qwen35.levels import PytestValidator, get_level
from gpumod.benchmarks.qwen35.llm_client import LlamaCppClient
from gpumod.benchmarks.qwen35.sampler_config import THINKING_CODING


def extract_code(response: str) -> str:
    if not response:
        return ""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            lines = parts[1].strip().split("\n")
            if lines and not lines[0].startswith(("def ", "class ", "import ", "from ")):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return response.strip()


def wilson_ci(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


async def run_one(client: LlamaCppClient, prompt: str, tests: str, max_tokens: int) -> dict:
    validator = PytestValidator(timeout_seconds=30)
    t0 = time.perf_counter()
    try:
        response = await client.generate(
            prompt, max_tokens=max_tokens, **THINKING_CODING.to_dict()
        )
    except Exception as exc:
        return {
            "passed": False,
            "error": f"generate raised: {exc}",
            "duration_seconds": time.perf_counter() - t0,
            "tests_passed": 0,
            "tests_total": 0,
        }
    duration = time.perf_counter() - t0
    reasoning = client.last_reasoning_content
    if "```" in response:
        source = response
    elif reasoning:
        source = reasoning + "\n\n" + response
    else:
        source = response
    code = extract_code(source)
    result = validator.validate(code, tests)
    return {
        "passed": result.passed,
        "tests_passed": result.tests_passed,
        "tests_total": result.tests_total,
        "error": result.error,
        "duration_seconds": duration,
        "tokens": (client.last_timing or {}).get("generated_tokens"),
    }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:7105")
    p.add_argument("--attempts", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/20260526_qwen36_mtp_vs_qwen35_heretic/l1_smoke.json"),
    )
    args = p.parse_args()

    level = get_level(1)
    client = LlamaCppClient(base_url=args.base_url, timeout=900.0)

    print(f"L1 ({level.name}) smoke vs {args.base_url} — {args.attempts} attempts")
    print("-" * 64)

    attempts: list[dict] = []
    try:
        for i in range(1, args.attempts + 1):
            row = await run_one(client, level.prompt, level.test_code, args.max_tokens)
            row["attempt"] = i
            row["timestamp"] = datetime.now(UTC).isoformat()
            attempts.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            tests = f"{row['tests_passed']}/{row['tests_total']}"
            err = f" [{row['error']}]" if row.get("error") and not row["passed"] else ""
            print(
                f"  #{i:02d}  {status}  tests={tests}  "
                f"{row['duration_seconds']:.1f}s  tokens={row.get('tokens')}{err}"
            )
    finally:
        await client.close()

    passes = sum(1 for r in attempts if r["passed"])
    n = len(attempts)
    rate = passes / n if n else 0.0
    lo, hi = wilson_ci(passes, n)
    mean_dur = sum(r["duration_seconds"] for r in attempts) / n if n else 0.0
    print("-" * 64)
    print(f"  pass rate : {passes}/{n} = {rate:.2%}  (Wilson 95% CI [{lo:.2%}, {hi:.2%}])")
    print(f"  mean time : {mean_dur:.1f}s / attempt")

    summary = {
        "base_url": args.base_url,
        "level": 1,
        "level_name": level.name,
        "attempts": attempts,
        "pass_rate": rate,
        "passes": passes,
        "n": n,
        "wilson_ci_95": [lo, hi],
        "mean_duration_seconds": mean_dur,
        "started_at": attempts[0]["timestamp"] if attempts else None,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"  saved     : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
