# Job Queue Challenge Benchmark

A graduated difficulty benchmark for evaluating LLM coding capabilities.

**Judge:** Claude Code (Opus 4.6) — designed prompts, ran benchmarks, scored results via pytest

----

## Overview

This benchmark tests an LLM's ability to implement increasingly complex features in a task queue system. Unlike simple pass/fail tests, it produces a **percentage score** that discriminates between model capabilities.

## Difficulty Levels

| Level | Task | Points | Observed Pass Rate |
|-------|------|--------|-------------------|
| L1 | Basic queue (add/get, FIFO) | 25 | 50% (2/4) |
| L2 | Retry with exponential backoff | 25 | 25% (1/4)* |
| L3 | Priority scheduling | 25 | 0% (0/4) |
| L4 | Find & fix concurrency bug | 15 | 50% (2/4) |
| L5 | Multi-file refactoring | 10 | 0% (0/4) |

*L2 requires precise retry timing; most models exhaust `max_tokens=8192` budget thinking before producing code.

**Total: 100 points**

## Score Interpretation

| Score | Interpretation |
|-------|----------------|
| 0-25 | Weak: Only basic operations work |
| 25-50 | Average: Basic + priority or concurrency |
| 50-75 | Good: Multiple advanced levels passed |
| 75-90 | Excellent: Most levels including L4 bug fix |
| 90-100 | Expert: Full refactoring capability |

## Running the Benchmark

### Prerequisites

```bash
# Ensure model is running (--parallel 1 for best quality)
uv run gpumod service start qwen35-35b-a3b-q3
```

### Run All Levels

```bash
uv run python docs/benchmarks/job_queue_challenge/benchmark_runner.py \
    --model qwen35-35b-a3b-q3 \
    --port 7091 \
    --output docs/benchmarks/job_queue_challenge/
```

### Run Specific Levels

```bash
# Only L1-L3
uv run python docs/benchmarks/job_queue_challenge/benchmark_runner.py \
    --model qwen35-35b-a3b-q3 \
    --port 7091 \
    --levels L1 L2 L3
```

## Test Details

### L1: Basic Queue Operations (5 tests)

- `add_job()` returns job_id
- `get_result()` returns computed value
- Multiple jobs execute correctly
- FIFO ordering maintained
- Nonexistent job handling

### L2: Retry with Backoff (5 tests)

- Job retries on exception
- Max 3 retries (4 total attempts)
- Exponential backoff: 1s, 2s, 4s
- Successful jobs don't retry
- Mixed success/failure handling

### L3: Priority Queue (5 tests)

- Higher priority executes first
- Same priority uses FIFO
- Mixed priorities sort correctly
- Default priority works
- Priority with args/kwargs

### L4: Concurrency Bug Fix (1 test)

Given buggy code with a race condition in `self.results[job_id] = result` (unprotected write), the model must:
1. Identify the bug
2. Fix it with proper locking
3. Pass concurrent completion test with 100 jobs

### L5: Multi-file Refactor (2 tests)

Refactor monolithic `queue.py` into:
```
queue/
  __init__.py    # Exports JobQueue
  core.py        # Base class
  retry.py       # Retry logic
  priority.py    # Priority handling
```

## Comparing Models

To compare models fairly:

1. **Same VRAM budget**: Compare models that fit in same memory
2. **Multiple runs**: Run 3x and average to account for variance
3. **Document architecture**: Note whether comparing MoE vs dense

### Recommended Comparisons

| Comparison | Models | Why Fair |
|------------|--------|----------|
| MoE vs Dense | 35B-A3B vs 27B | Different architectures, similar total params |
| Quantization impact | Q4 vs Q3 of same model | Isolates quant quality |
| Architecture + Size | 35B-A3B Q3 vs 27B Q4 | Similar VRAM footprint |

## Output Format

```json
{
  "model_id": "qwen35-27b-q3",
  "model_name": "qwen35-27b-q3",
  "timestamp": "2026-02-26T06:28:27.540064+00:00",
  "total_duration_ms": 487251,
  "scores": {
    "L1": 25,
    "L2": 25,
    "L3": 0,
    "L4": 15,
    "L5": 0,
    "total": 65,
    "percentage": 65.0
  },
  "levels": [...]
}
```

## Benchmark Results (2026-02-26)

### Configuration

```bash
# Single-slot mode (--parallel 1) for maximum quality per request
# llama.cpp preset: --parallel 1 --threads 16 (no cont-batching)
# Benchmark runner: 1 request at a time, max_tokens=8192, temperature=0.1

uv run python docs/benchmarks/job_queue_challenge/benchmark_runner.py \
    --model qwen35-27b-q3 \
    --port 7093 \
    --output docs/benchmarks/job_queue_challenge/
```

**Hardware:** RTX 4090 (24GB VRAM)
**llama.cpp flags:**
- `--parallel 1` — Single concurrent request (no batching)
- `--threads 16` — CPU thread count
- `--jinja` — Enable Jinja chat templates (required for Qwen3.5)
- `-ngl -1` — Full GPU offload

**Benchmark settings:**
- `max_tokens=8192` — Token generation limit
- `temperature=0.1` — Low temperature for deterministic output
- `/no_think` prefix — Disable chain-of-thought for direct code output

### Summary

| Model | Total | L1 | L2 | L3 | L4 | L5 | Time |
|-------|-------|----|----|----|----|----| -----|
| **Qwen3.5-27B Q3** | **65%** | 25 | **25** | 0 | **15** | 0 | 443s |
| Qwen3.5-35B-A3B Q3 | 25% | 25 | 0 | 0 | 0 | 0 | 243s |
| Qwen3.5-27B Q4 | 15% | 0 | 0 | 0 | **15** | 0 | 778s |
| Qwen3.5-35B-A3B Q4 | 0% | 0 | 0 | 0 | 0 | 0 | 263s |

### Key Findings

1. **27B Q3 passed L2 (retry logic)** — First model to successfully implement exponential backoff with correct timing
2. **High variance across runs** — Results differ significantly between benchmark runs; models are sensitive to prompt/context
3. **Dense 27B more reliable** — Dense architecture showed more consistent output vs MoE models with empty responses
4. **MoE 35B-A3B faster but less reliable** — 243s vs 443s, but more empty responses (timeouts)
5. **Empty responses common** — Thinking models often exhaust token budget before producing code

### Architecture Comparison

| Aspect | 27B (Dense) | 35B-A3B (MoE) |
|--------|-------------|---------------|
| Active params | 27B | 3B |
| L2 Retry Logic | ✅ Q3 passes | ❌ All fail |
| L4 Bug Fix | Q3/Q4 mixed | ❌ All fail this run |
| Speed | Slower (80-200s per level) | Faster (30-60s per level) |
| Best score | 65% (Q3) | 25% (Q3) |

### Variance Warning

These results are from a single run. Benchmark variance is high due to:
- Non-deterministic LLM sampling (even at temperature=0.1)
- Thinking models exhausting token budget before output
- `/no_think` effectiveness varies by prompt

For reliable comparisons, run 3+ iterations and average.

## Design Philosophy

1. **Graduated difficulty**: Not all models should pass all levels
2. **Automatic scoring**: No subjective evaluation
3. **Real-world task**: Task queues are common in production
4. **Discriminating**: Score spread should reveal capability differences

## Files

| File | Description |
|------|-------------|
| `benchmark_runner.py` | Main benchmark script |
| `test_job_queue.py` | pytest test suite |
| `result_*.json` | Benchmark results |
| `artifacts/<model>/` | Generated code from each level |
| `README.md` | This file |

### Artifacts

The benchmark runner saves all generated code to `artifacts/<model>/`:
- `L1_job_queue.py` — Basic queue implementation
- `L2_job_queue.py` — Queue with retry logic
- `L3_job_queue.py` — Queue with priority scheduling
- `L4_job_queue.py` — Fixed concurrency bug
- `L5_job_queue.py` — Multi-file refactor response

These artifacts allow inspection of model output and reproduction of results.
