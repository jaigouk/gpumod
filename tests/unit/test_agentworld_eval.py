"""Unit tests for the AgentWorldBench eval core (gpumod-kpmq.1).

Covers the deterministic foundation: sample model (history/ground-truth slicing),
prompt assembly (system + interleaved trajectory + current action), and score
aggregation. The Claude judge (live) and dataset download are integration-tested
elsewhere.
"""

from __future__ import annotations

import pytest

from gpumod.benchmarks.agentworld.dataset import take_per_domain
from gpumod.benchmarks.agentworld.judge import build_judge_prompt, parse_judge_response
from gpumod.benchmarks.agentworld.models import AgentWorldSample
from gpumod.benchmarks.agentworld.prompt import build_messages
from gpumod.benchmarks.agentworld.scorer import DimensionScores, aggregate

# Compact fixture matching the real Qwen/AgentWorldBench schema (turn_idx is 1-indexed;
# prompt[i] -> response[i] are turns 1..turn_idx; the turn to predict is turn_idx).
_ROW = {
    "task": "terminal",
    "id": 7,
    "system_str": "You are a Terminal World Model.",
    "prompt": ["### Turn 1\npwd", "### Turn 2\nls", "### Turn 3\ncat a.txt"],
    "response": ["/home/u", "a.txt  b.txt", "hello"],
    "current_prompt": "### Turn 3\ncat a.txt",
    "turn_idx": 3,
    "total_turns": 5,
}


def test_sample_from_row_history_and_ground_truth() -> None:
    s = AgentWorldSample.from_row(_ROW)
    assert s.task == "terminal"
    assert s.turn_idx == 3
    # ground truth = the observation for the current turn (turn_idx-1 index)
    assert s.ground_truth == "hello"
    # history = turns 1..turn_idx-1 as (action, observation) pairs
    assert s.history == [("### Turn 1\npwd", "/home/u"), ("### Turn 2\nls", "a.txt  b.txt")]


def test_build_messages_interleaves_history_then_current_action() -> None:
    s = AgentWorldSample.from_row(_ROW)
    msgs = build_messages(s)
    assert msgs[0] == {"role": "system", "content": "You are a Terminal World Model."}
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    # last message is the current action, NOT the ground-truth observation
    assert msgs[-1] == {"role": "user", "content": "### Turn 3\ncat a.txt"}
    assert all(m["content"] != "hello" for m in msgs)  # ground truth never leaks into the prompt


def test_build_messages_first_turn_has_no_history() -> None:
    row = {
        **_ROW,
        "prompt": ["### Turn 1\npwd"],
        "response": ["/home/u"],
        "current_prompt": "### Turn 1\npwd",
        "turn_idx": 1,
    }
    msgs = build_messages(AgentWorldSample.from_row(row))
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[-1]["content"] == "### Turn 1\npwd"


def test_dimension_score_normalization() -> None:
    assert DimensionScores(1, 1, 1, 1, 1).score_0_100() == 0.0  # all 1 -> 0
    assert DimensionScores(5, 5, 5, 5, 5).score_0_100() == 100.0  # all 5 -> 100
    assert DimensionScores(3, 3, 3, 3, 3).score_0_100() == 50.0  # mid -> 50


def test_aggregate_per_domain_and_overall() -> None:
    scored = [
        ("terminal", DimensionScores(5, 5, 5, 5, 5)),  # 100
        ("terminal", DimensionScores(3, 3, 3, 3, 3)),  # 50
        ("android", DimensionScores(1, 1, 1, 1, 1)),  # 0
    ]
    agg = aggregate(scored)
    assert agg["overall"]["mean"] == 50.0  # (100+50+0)/3
    assert agg["overall"]["n"] == 3
    assert agg["per_domain"]["terminal"]["mean"] == 75.0
    assert agg["per_domain"]["terminal"]["n"] == 2
    assert agg["per_domain"]["android"]["mean"] == 0.0


def test_parse_judge_response_plain_json() -> None:
    ds = parse_judge_response(
        '{"format":5,"factuality":4,"consistency":3,"realism":2,"quality":1,"rationale":"x"}'
    )
    assert (ds.format, ds.factuality, ds.consistency, ds.realism, ds.quality) == (5, 4, 3, 2, 1)


def test_parse_judge_response_strips_fences_and_prose() -> None:
    txt = (
        "Grade:\n```json\n"
        '{"format":3,"factuality":3,"consistency":3,"realism":3,"quality":3}\n'
        "```\nok"
    )
    assert parse_judge_response(txt).mean_raw() == 3.0


def test_parse_judge_response_clamps_out_of_range() -> None:
    ds = parse_judge_response(
        '{"format":9,"factuality":0,"consistency":3,"realism":3,"quality":3}'
    )
    assert ds.format == 5  # 9 -> 5
    assert ds.factuality == 1  # 0 -> 1


def test_parse_judge_response_missing_dim_raises() -> None:
    with pytest.raises(ValueError, match="missing dimension"):
        parse_judge_response('{"format":3,"factuality":3}')


def test_build_judge_prompt_includes_truth_pred_and_dims() -> None:
    p = build_judge_prompt("terminal", prediction="PRED_X", ground_truth="TRUTH_Y")
    assert "PRED_X" in p
    assert "TRUTH_Y" in p
    for dim in ("format", "factuality", "consistency", "realism", "quality"):
        assert dim in p.lower()


def test_take_per_domain_groups_and_limits() -> None:
    rows = [
        {"task": "terminal"},
        {"task": "terminal"},
        {"task": "terminal"},
        {"task": "android"},
        {"task": "android"},
    ]
    out = take_per_domain(rows, 2)
    tasks = [r["task"] for r in out]
    assert tasks.count("terminal") == 2
    assert tasks.count("android") == 2
    assert len(take_per_domain(rows, None)) == 5  # None -> all
