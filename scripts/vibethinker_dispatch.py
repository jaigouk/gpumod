#!/usr/bin/env python3
"""Dispatch a reasoning task to the VibeThinker-3B multi-slot service (:7115).

VibeThinker is a high-variance (sigma 15.45), verifiable single-shot reasoner —
NOT a tool-calling / agent / composition model (gpumod-msy8 bench + model card).
So this helper does two things the model demands:

  1. ROLES as system prompts. tech-lead / researcher / developer / QA are the
     same model with a different persona — there is no reason to run separate
     processes (that would re-load the 3.3 GB weights per role in VRAM). One
     `--parallel 4` service (vibethinker-3b-q8-multi4) serves all four
     concurrently; the role is just the system prompt sent per request.

  2. BEST-OF-N. `--n` fires N concurrent samples of the same task at the slots,
     so a verifier can pick the best (or majority-vote). Never trust a single
     VibeThinker generation.

It strips closed <think>...</think> spans before returning, because VibeThinker
emits reasoning into message.content (its plain Qwen2 ChatML is content-only in
llama.cpp, so reasoning_content stays empty — no server flag splits it).

Examples:
    # best-of-4 on a verifiable math problem, generous depth
    python scripts/vibethinker_dispatch.py --role developer --n 4 --max-tokens 40000 \\
        "Implement an O(n log n) algorithm for the longest increasing subsequence."

    # single QA pass over a design, read prompt from stdin
    echo "Review this retry policy for race conditions: ..." | \\
        python scripts/vibethinker_dispatch.py --role qa
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

DEFAULT_URL = "http://localhost:7115/v1/chat/completions"

# Role personas. VibeThinker's strength is verifiable, self-contained reasoning,
# so each persona leans into "reason rigorously, then commit to a final answer".
ROLES: dict[str, str] = {
    "tech-lead": (
        "You are a staff engineer doing system design and review. Reason rigorously "
        "about correctness, failure modes, and trade-offs inside <think>, then give a "
        "DECISIVE, justified recommendation. Communicate fully but efficiently — short "
        "labeled sections, no filler. When you assign work, scope each task small enough "
        "to finish in one pass and specify it completely: signatures, inputs, outputs, "
        "a worked example."
    ),
    "researcher": (
        "You are a research analyst. Reason from first principles and ONLY the facts "
        "given — never invent sources. Think inside <think>, then give a structured "
        "synthesis with explicit assumptions, noting which given fact supports each "
        "claim. Thorough but not verbose."
    ),
    "developer": (
        "You are an expert programmer who works TEST-FIRST (TDD: write failing tests, then "
        "minimal code to pass, then refactor) in a SOLID, domain-driven (DDD) style — model "
        "the domain explicitly, single-responsibility modules, depend on abstractions. Ship "
        "COMPLETE, runnable code — never stubs, '...', or truncation. Your code is checked by "
        "REAL ruff, type-check, and pytest; write to pass them, and when given gate errors fix "
        "them precisely and return the FULL corrected files. Reason inside <think> but leave "
        "room to emit the full code after </think>. Output each file as a line 'FILE: <path>' "
        "followed by one fenced code block; don't narrate between files."
    ),
    "qa": (
        "You are a QA engineer reviewing code that has ALREADY passed lint/type-check/tests — "
        "your job is correctness and design, not styling. Verify it meets every acceptance "
        "criterion and the DDD/SOLID intent, and hunt edge cases the tests miss (empty/missing "
        "input, unicode, malformed data, idempotency, boundary counts). You cannot execute; "
        "reason inside <think>. Give a DECISIVE verdict (PASS only if it truly meets every "
        "criterion, else FAIL); per issue give the file, the symptom, and a concrete "
        "repro/failing input. Terse and labeled."
    ),
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Sample:
    index: int
    latency_s: float
    completion_tokens: int | None
    finish_reason: str | None
    answer: str


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def call_once(index: int, messages: list[dict[str, str]], args: argparse.Namespace) -> Sample:
    payload = {
        "model": "vibethinker",
        "messages": messages,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    data = json.dumps(payload).encode()
    # S310 suppressed: args.url is an operator-supplied local llama-server endpoint.
    req = urllib.request.Request(  # noqa: S310
        args.url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:  # noqa: S310
        body = json.loads(resp.read())
    dt = time.time() - t0
    choice = body["choices"][0]
    return Sample(
        index=index,
        latency_s=dt,
        completion_tokens=body.get("usage", {}).get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        answer=strip_think(choice["message"]["content"]),
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return " ".join(args.prompt)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            return fh.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide a prompt (positional, --file, or stdin)")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("prompt", nargs="*", help="task text (or use --file / stdin)")
    p.add_argument("--role", choices=sorted(ROLES), help="persona system prompt")
    p.add_argument("--file", help="read the task from this file")
    p.add_argument("-n", "--n", type=int, default=1, help="best-of-N concurrent samples")
    p.add_argument("--max-tokens", type=int, default=16384, help="per-request generation cap")
    p.add_argument("--temp", type=float, default=1.0, help="card default 1.0")
    p.add_argument("--top-p", type=float, default=0.95, help="card default 0.95")
    p.add_argument("--top-k", type=int, default=0, help="card default disabled (0)")
    p.add_argument("--timeout", type=float, default=1800.0, help="client timeout (s)")
    p.add_argument("--url", default=DEFAULT_URL)
    return p.parse_args()


def main() -> None:
    args = build_args()
    task = read_prompt(args)

    messages: list[dict[str, str]] = []
    if args.role:
        messages.append({"role": "system", "content": ROLES[args.role]})
    messages.append({"role": "user", "content": task})

    label = args.role or "plain"
    print(f"[{label}] firing {args.n} sample(s) at {args.url}\n", file=sys.stderr)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        samples = list(ex.map(lambda i: call_once(i, messages, args), range(args.n)))
    wall = time.time() - t0

    for s in sorted(samples, key=lambda x: x.index):
        truncated = " (TRUNCATED at max_tokens)" if s.finish_reason == "length" else ""
        print(
            f"===== sample {s.index + 1}/{args.n} | {s.latency_s:.1f}s | "
            f"{s.completion_tokens} tok{truncated} ====="
        )
        print(s.answer)
        print()

    if args.n > 1:
        print(
            f"--- best-of-{args.n}: wall {wall:.1f}s; verify the {args.n} answers "
            f"above and keep the best ---",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
