#!/usr/bin/env python
"""Run the AgentWorldBench world-modeling eval (gpumod-kpmq.1).

Two-phase, per sample: the model-under-test (served via gpumod, OpenAI-compatible)
predicts the next environment observation; Claude (`claude -p`, subscription, no API key)
judges it against the ground truth on five dimensions. Aggregates per-domain + overall
and saves a result JSON.

Prereqs: the model is already serving (e.g. `gpumod service start gemma4-e2b-qat-q2`)
and the `claude` CLI is logged in (`claude login`). Example:

  uv run python scripts/run_agentworld_benchmark.py \
      --model-name gemma4-e2b-qat-q2 --base-url http://localhost:7112/v1 \
      --sample 2 --output-dir docs/benchmarks/20260625_agentworld_worldmodel/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

from gpumod.benchmarks.agentworld.dataset import DOMAINS, load_samples
from gpumod.benchmarks.agentworld.judge import ClaudeJudge
from gpumod.benchmarks.agentworld.prompt import build_messages
from gpumod.benchmarks.agentworld.scorer import DimensionScores, aggregate
from gpumod.benchmarks.normalizers import TextAnswerNormalizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentWorldBench world-modeling eval")
    p.add_argument("--model-name", required=True, help="label for the model-under-test")
    p.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible URL of the served model, e.g. http://localhost:7112/v1",
    )
    p.add_argument("--model", default="local", help="model name sent to the served endpoint")
    p.add_argument(
        "--sample", type=int, default=None, help="max samples per domain (default: all)"
    )
    p.add_argument("--domains", nargs="*", default=None, help=f"subset of {DOMAINS}")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--judge-model",
        default=None,
        help="claude -p --model (default: opus / CLAUDE_JUDGE_MODEL)",
    )
    p.add_argument("--gen-temp", type=float, default=0.6)
    p.add_argument("--gen-max-tokens", type=int, default=2048)
    p.add_argument("--gen-timeout", type=float, default=300.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains = tuple(args.domains) if args.domains else None

    samples = load_samples(domains=domains, sample_per_domain=args.sample)
    seen = sorted({s.task for s in samples})
    print(f"Loaded {len(samples)} samples across {len(seen)} domains: {seen}")

    gen = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=args.gen_timeout)
    judge = ClaudeJudge(model=args.judge_model)
    # gpumod-nor9: strip <think> traces / route reasoning_content so reasoning
    # models are scored on their final prediction, not the raw CoT dump.
    normalizer = TextAnswerNormalizer()

    scored: list[tuple[str, DimensionScores]] = []
    records: list[dict] = []
    for i, s in enumerate(samples, 1):
        messages = build_messages(s)
        t0 = time.time()
        try:
            resp = gen.chat.completions.create(
                model=args.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=args.gen_max_tokens,
                temperature=args.gen_temp,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            # openai SDK exposes non-standard fields (reasoning_content) via
            # model_extra/getattr; absent -> None -> normalizer degrades to
            # stripped content.
            reasoning = getattr(msg, "reasoning_content", None)
            prediction = normalizer.extract_answer(content, reasoning)
        except Exception as exc:
            print(f"[{i}/{len(samples)}] {s.task} GEN ERROR: {exc}")
            continue
        gen_s = time.time() - t0
        try:
            ds = judge.score(s.task, prediction, s.ground_truth)
        except Exception as exc:
            print(f"[{i}/{len(samples)}] {s.task} JUDGE ERROR: {exc}")
            continue

        scored.append((s.task, ds))
        records.append(
            {
                "task": s.task,
                "id": s.id,
                "turn_idx": s.turn_idx,
                "score_0_100": ds.score_0_100(),
                "dimensions": asdict(ds),
                "gen_seconds": round(gen_s, 2),
                "prediction": prediction,
                "ground_truth": s.ground_truth,
            }
        )
        print(f"[{i}/{len(samples)}] {s.task:9} score={ds.score_0_100():.1f}  ({gen_s:.1f}s gen)")

    summary = aggregate(scored)
    result = {
        "model_name": args.model_name,
        "base_url": args.base_url,
        "judge": {
            "via": "claude -p",
            "model": args.judge_model or "opus",
        },
        "sample_per_domain": args.sample,
        "n_scored": len(scored),
        "n_loaded": len(samples),
        "timestamp": datetime.now(UTC).isoformat(),
        "summary": summary,
        "records": records,
    }
    out = args.output_dir / f"result_agentworld_{args.model_name}.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\nSaved {out}")
    if scored:
        print(f"Overall: {summary['overall']['mean']:.1f}/100 (n={summary['overall']['n']})")
        for dom, st in sorted(summary["per_domain"].items()):
            print(f"  {dom:9} {st['mean']:.1f}  (n={st['n']})")
    else:
        print("No samples scored (all gen/judge calls failed) — check the model + `claude -p`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
