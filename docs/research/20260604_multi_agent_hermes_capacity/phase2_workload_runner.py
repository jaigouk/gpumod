#!/usr/bin/env python3
"""Phase 2 workload runner for gpumod-8xaq.

Spawns N concurrent "agent" threads against a running llama-server
OpenAI-compatible /v1/chat/completions endpoint. Each agent follows a
configurable role profile (request cadence, max_tokens, simulated tool
result size, reasoning on/off). Collects per-request and per-agent
metrics, emits a JSON results blob on stdout when the duration elapses.

Synthetic tool overhead: each turn prepends a fake "[TOOL RESULT]" blob
of N tokens of repetitive code-like text to the user message. This
exercises the same KV-cache pressure path as real tool use without
requiring a tool executor in the loop.

Self-contained: stdlib only (threading + urllib + json + time + random).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Role profiles
# ---------------------------------------------------------------------------


@dataclass
class RoleProfile:
    name: str
    cadence_secs: float  # 0 = continuous (next request fires when previous returns)
    max_tokens: int
    tool_overhead_tokens: int  # simulated [TOOL RESULT] blob size
    enable_thinking: bool
    user_prompt: str


PROFILES = {
    "TL": RoleProfile(
        name="TL",
        cadence_secs=120.0,  # 1 req / 2 min, bursty
        max_tokens=1500,
        tool_overhead_tokens=400,
        enable_thinking=True,
        user_prompt=(
            "Review this design proposal at a high level. Identify the top 2 risks "
            "and one alternative approach worth considering. Be concise."
        ),
    ),
    "Dev": RoleProfile(
        name="Dev",
        cadence_secs=0.0,  # continuous
        max_tokens=8000,
        tool_overhead_tokens=15000,
        enable_thinking=True,
        user_prompt=(
            "Implement the requested function. Walk through your reasoning, write "
            "the code, then explain how you'd test it. Refer to the provided file "
            "contents above."
        ),
    ),
    "QA": RoleProfile(
        name="QA",
        cadence_secs=60.0,  # 1 req / 1 min
        max_tokens=2500,
        tool_overhead_tokens=5000,
        enable_thinking=True,
        user_prompt=(
            "Analyse the test failure shown above. Identify the root cause and "
            "suggest a minimal fix plus a regression test."
        ),
    ),
    "Research": RoleProfile(
        name="Research",
        cadence_secs=0.0,  # continuous
        max_tokens=5000,
        tool_overhead_tokens=8000,
        enable_thinking=True,
        user_prompt=(
            "Synthesise the source material above into a structured analysis. Cover "
            "the main claims, the evidence quality, and any conflicting findings."
        ),
    ),
    "ToolStorm": RoleProfile(
        name="ToolStorm",
        cadence_secs=0.0,  # continuous, rapid
        max_tokens=500,  # small responses
        tool_overhead_tokens=800,  # small but frequent tool results
        enable_thinking=False,  # skip reasoning to keep storm fast
        user_prompt="Acknowledge receipt and call the next tool.",
    ),
}


# ---------------------------------------------------------------------------
# Tool overhead simulation
# ---------------------------------------------------------------------------


def _build_filler(target_tokens: int, seed: int = 0) -> str:
    """Build a chunk of code-like text whose tokenised length is roughly the
    target. We approximate at 4 chars / token (English / code mix).
    """
    rng = random.Random(seed)
    target_chars = target_tokens * 4
    lines = []
    builders = [
        lambda: f"def {''.join(rng.choices(string.ascii_lowercase, k=8))}(arg_{rng.randint(0, 99)}: int) -> bool:",
        lambda: f"    return arg_{rng.randint(0, 99)} > {rng.randint(1, 999)}",
        lambda: f"# TODO({''.join(rng.choices(string.ascii_lowercase, k=5))}): {''.join(rng.choices(string.ascii_lowercase + ' ', k=40))}",
        lambda: f"        self.{''.join(rng.choices(string.ascii_lowercase, k=8))} = {{'k_{rng.randint(0, 9)}': {rng.randint(0, 9999)}}}",
        lambda: f"        log.info('processing batch %d with %d items', {rng.randint(0, 999)}, {rng.randint(0, 999)})",
        lambda: f"        raise ValueError(f'invalid {{state.{''.join(rng.choices(string.ascii_lowercase, k=4))}}}')",
        "",
    ]
    while sum(len(line) + 1 for line in lines) < target_chars:
        b = rng.choice(builders)
        lines.append(b() if callable(b) else b)
    return "\n".join(lines)[:target_chars]


def build_user_content(profile: RoleProfile, turn_index: int) -> str:
    if profile.tool_overhead_tokens > 0:
        filler = _build_filler(profile.tool_overhead_tokens, seed=turn_index)
        return (
            f"[TOOL RESULT — turn {turn_index}, simulated ~{profile.tool_overhead_tokens} tokens of context]\n"
            f"{filler}\n"
            f"[/TOOL RESULT]\n\n"
            f"{profile.user_prompt}"
        )
    return profile.user_prompt


# ---------------------------------------------------------------------------
# Per-request execution
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    agent: str
    turn_index: int
    started_at: float  # epoch seconds
    duration_secs: float  # wall-clock from POST to full response
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    http_status: int
    error: str | None = None

    @property
    def tps(self) -> float:
        if self.duration_secs <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / self.duration_secs


def execute_turn(
    base_url: str,
    model: str,
    agent: str,
    turn_index: int,
    profile: RoleProfile,
    timeout: float = 600.0,
) -> TurnResult:
    content = build_user_content(profile, turn_index)
    payload: dict = {
        "model": model,
        "max_tokens": profile.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if not profile.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    started = time.time()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        duration = time.time() - started
        usage = data.get("usage", {})
        choice = (data.get("choices") or [{}])[0]
        return TurnResult(
            agent=agent,
            turn_index=turn_index,
            started_at=started,
            duration_secs=duration,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "?"),
            http_status=200,
        )
    except urllib.error.HTTPError as e:
        duration = time.time() - started
        return TurnResult(
            agent=agent,
            turn_index=turn_index,
            started_at=started,
            duration_secs=duration,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="error",
            http_status=e.code,
            error=str(e),
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        duration = time.time() - started
        return TurnResult(
            agent=agent,
            turn_index=turn_index,
            started_at=started,
            duration_secs=duration,
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="error",
            http_status=0,
            error=type(e).__name__ + ": " + str(e),
        )


# ---------------------------------------------------------------------------
# Agent threads + orchestration
# ---------------------------------------------------------------------------


@dataclass
class AgentSlot:
    label: str  # e.g. "TL-1", "Dev-1", "Research-2"
    profile: RoleProfile


def agent_loop(
    slot: AgentSlot,
    base_url: str,
    model: str,
    stop_at: float,
    results: list[TurnResult],
    results_lock: threading.Lock,
    print_lock: threading.Lock,
) -> None:
    turn_index = 0
    while time.time() < stop_at:
        result = execute_turn(base_url, model, slot.label, turn_index, slot.profile)
        with results_lock:
            results.append(result)
        with print_lock:
            err = f" ERR={result.error[:60]}" if result.error else ""
            sys.stdout.write(
                f"  [{slot.label} turn {turn_index}] {result.duration_secs:6.1f}s  "
                f"{result.completion_tokens:>5} tok  {result.tps:5.1f} TPS  "
                f"finish={result.finish_reason}  http={result.http_status}{err}\n"
            )
            sys.stdout.flush()
        turn_index += 1
        if slot.profile.cadence_secs > 0 and time.time() < stop_at:
            # Sleep the cadence MINUS the time we already spent on the call
            # so cadence-driven roles fire at their intended rate even when
            # turns finish quickly.
            sleep_left = slot.profile.cadence_secs - result.duration_secs
            if sleep_left > 0:
                time.sleep(min(sleep_left, stop_at - time.time()))


def run_workload(
    slots: list[AgentSlot],
    base_url: str,
    model: str,
    duration_secs: float,
) -> dict:
    started_at = time.time()
    stop_at = started_at + duration_secs

    results: list[TurnResult] = []
    results_lock = threading.Lock()
    print_lock = threading.Lock()

    threads = [
        threading.Thread(
            target=agent_loop,
            args=(slot, base_url, model, stop_at, results, results_lock, print_lock),
            name=f"agent-{slot.label}",
            daemon=True,
        )
        for slot in slots
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=duration_secs + 60.0)

    completed_at = time.time()
    return summarise(slots, results, started_at, completed_at, duration_secs)


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarise(
    slots: list[AgentSlot],
    results: list[TurnResult],
    started_at: float,
    completed_at: float,
    duration_secs: float,
) -> dict:
    by_agent: dict[str, list[TurnResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent, []).append(r)

    per_agent_stats = {}
    for slot in slots:
        agent_results = by_agent.get(slot.label, [])
        successful = [r for r in agent_results if r.http_status == 200]
        if successful:
            durations = [r.duration_secs for r in successful]
            tokens = [r.completion_tokens for r in successful]
            tps_per_call = [r.tps for r in successful if r.tps > 0]
            per_agent_stats[slot.label] = {
                "profile": asdict(slot.profile),
                "turns_total": len(agent_results),
                "turns_ok": len(successful),
                "turns_err": len(agent_results) - len(successful),
                "tokens_total": sum(tokens),
                "tokens_per_turn_mean": statistics.mean(tokens),
                "turn_duration_p50": _pct(durations, 50),
                "turn_duration_p95": _pct(durations, 95),
                "tps_per_call_mean": statistics.mean(tps_per_call) if tps_per_call else 0.0,
                "throughput_tps_window": sum(tokens) / duration_secs,
            }
        else:
            per_agent_stats[slot.label] = {
                "profile": asdict(slot.profile),
                "turns_total": len(agent_results),
                "turns_ok": 0,
                "turns_err": len(agent_results),
                "tokens_total": 0,
                "errors": [r.error for r in agent_results[:3]],
            }

    aggregate_tokens = sum(r.completion_tokens for r in results if r.http_status == 200)
    return {
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_secs": duration_secs,
        "n_agents": len(slots),
        "per_agent": per_agent_stats,
        "aggregate_tokens": aggregate_tokens,
        "aggregate_tps_window": aggregate_tokens / duration_secs,
        "turns_total": len(results),
        "turns_ok": sum(1 for r in results if r.http_status == 200),
        "turns_err": sum(1 for r in results if r.http_status != 200),
        "raw_turns": [asdict(r) for r in results],
    }


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_slots(spec: str) -> list[AgentSlot]:
    """Parse a role-spec string like 'TL,Dev,QA' or '2xResearch,1xDev'."""
    slots: list[AgentSlot] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "x" in chunk:
            count_str, role = chunk.split("x", 1)
            count = int(count_str)
        else:
            count, role = 1, chunk
        role = role.strip()
        if role not in PROFILES:
            raise SystemExit(f"unknown role: {role!r}. Available: {list(PROFILES)}")
        for i in range(count):
            label = f"{role}-{i + 1}" if count > 1 else role
            slots.append(AgentSlot(label=label, profile=PROFILES[role]))
    return slots


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:7109", help="llama-server base URL")
    p.add_argument("--model", default="gemma-4-26B-A4B", help="model id (ignored by llama.cpp, kept for OpenAI compat)")
    p.add_argument("--duration", type=float, default=600.0, help="seconds to run")
    p.add_argument("--slots", required=True, help="comma-separated role spec, e.g. 'TL,Dev,QA' or '3xResearch'")
    p.add_argument("--output", default="-", help="write JSON summary here (use - for stdout)")
    p.add_argument("--config-label", default="", help="label written into summary for cross-run aggregation")
    args = p.parse_args()

    slots = parse_slots(args.slots)
    print(f"=== Workload: {args.slots} ({len(slots)} slots) | duration {args.duration:.0f}s ===")
    for s in slots:
        print(
            f"  {s.label:<14}  cadence={s.profile.cadence_secs:>5.0f}s  "
            f"max_tokens={s.profile.max_tokens:<5}  "
            f"tool_overhead={s.profile.tool_overhead_tokens:<6}  "
            f"thinking={s.profile.enable_thinking}"
        )

    summary = run_workload(slots, args.base_url, args.model, args.duration)
    summary["config_label"] = args.config_label

    if args.output == "-":
        json.dump(summary, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"=== summary written to {args.output} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
