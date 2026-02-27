# Qwen3.5 Job Queue Challenge Benchmark

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

### Run All Levels (Single Iteration)

```bash
uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
    --model qwen35-35b-a3b-q3 \
    --port 7091 \
    --output docs/benchmarks/20260226_qwen35_job_queue_challenge/
```

### Run with Multiple Iterations (Recommended)

For reliable results, run 5 iterations:

```bash
uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
    --model qwen35-35b-a3b-q3 \
    --port 7091 \
    --iterations 5 \
    --output docs/benchmarks/20260226_qwen35_job_queue_challenge/
```

This produces statistics (best, avg, min, max) and saves artifacts from the best-scoring run.

### Run Specific Levels

```bash
# Only L1-L3
uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
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
  "port": 7093,
  "timestamp": "2026-02-26T06:28:27.540064+00:00",
  "iterations": 5,
  "total_duration_ms": 2436255,
  "statistics": {
    "best_score": 65,
    "avg_score": 45.0,
    "min_score": 25,
    "max_score": 65,
    "all_scores": [25, 40, 65, 40, 55],
    "best_iteration": 3
  },
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

The `statistics` block is always included. Artifacts are saved from the best-scoring iteration.

## Benchmark Results (2026-02-26)

### Configuration

```bash
# Single-slot mode (--parallel 1) for maximum quality per request
# llama.cpp preset: --parallel 1 --threads 16 (no cont-batching)
# Benchmark runner: 5 iterations, max_tokens=8192, temperature=0.1

uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
    --model qwen35-27b-q3 \
    --port 7093 \
    --iterations 5 \
    --output docs/benchmarks/20260226_qwen35_job_queue_challenge/
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
- `iterations=5` — Multiple runs for statistical reliability

### Summary (5 Iterations)

| Model | Best | Avg | L1 | L2 | L3 | L4 | L5 | Scores | Time |
|-------|------|-----|----|----|----|----|----| ------ |------|
| **Qwen3.5-27B Q3** | **90%** | 62% | 25 | 25 | 25 | 15 | 0 | [65,45,65,90,45] | 45min |
| Qwen3.5-27B Q4 | 65% | 45% | 25 | 25 | 0 | 15 | 0 | [65,40,40,15,65] | 60min |
| Qwen3.5-35B-A3B Q3 | 65% | 25% | 25 | 0 | 25 | 15 | 0 | [25,0,20,65,15] | 21min |
| Qwen3.5-35B-A3B Q4* | 65% | 42% | 25 | 0 | 25 | 15 | 0 | [20,40,65] | 13min |

*Q4 MoE ran 3 iterations (benchmark stuck on L2 retry tests)

### Key Findings

1. **27B Q3 achieved 90% peak score** — First model to pass L1+L2+L3+L4, only missing L5 multi-file refactoring
2. **Extreme variance** — Same model scored 45-90% across runs; 35B-A3B Q3 ranged 0-65%
3. **Dense 27B more consistent** — 27B Q3 avg 62% vs MoE 35B-A3B avg 25% despite same best score (65%)
4. **MoE models faster but less reliable** — 2-3x faster but higher variance and more empty responses
5. **L5 refactoring unsolved** — No model successfully completed multi-file refactoring across any iteration
6. **Q3 quant outperforms Q4** — 27B Q3 (90% best) > 27B Q4 (65% best) despite lower precision

### Architecture Comparison

| Aspect | 27B (Dense) | 35B-A3B (MoE) |
|--------|-------------|---------------|
| Active params | 27B | 3B |
| Best score | 90% (Q3) | 65% (Q3/Q4) |
| Average score | 62% Q3, 45% Q4 | 25% Q3, 42% Q4 |
| L2 Retry Logic | ✅ Both pass | ❌ Both fail |
| L3 Priority | Q3 passes | Both pass (best run) |
| L4 Bug Fix | Both pass | Both pass (best run) |
| Speed | Slower (45-60min) | Faster (13-21min) |
| Consistency | High (45-90 range) | Low (0-65 range) |

### Variance Analysis

Multi-iteration runs revealed significant score variance:

| Model | Min | Max | Spread | Std Dev |
|-------|-----|-----|--------|---------|
| 27B Q3 | 45 | 90 | 45 | ~18 |
| 27B Q4 | 15 | 65 | 50 | ~21 |
| 35B-A3B Q3 | 0 | 65 | 65 | ~24 |
| 35B-A3B Q4 | 20 | 65 | 45 | ~19 |

**Causes:**
- Non-deterministic LLM sampling (even at temperature=0.1)
- Thinking models exhaust token budget before producing code
- `/no_think` effectiveness varies by prompt and context
- MoE routing decisions can vary between runs

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
