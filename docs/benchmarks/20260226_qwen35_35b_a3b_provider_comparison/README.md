# Qwen3.5-35B-A3B IQ4 Provider Comparison

**Date:** 2026-02-27 (v1), 2026-03-01 (v2)

## Goal

Compare IQ4-class GGUF quantizations from different providers on speed, VRAM, and coding task performance.

## Background

KLD/PPL rankings for Qwen3.5-35B-A3B quantizations already exist ([Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/)). This benchmark adds:

1. **TPS (tokens per second)** — Speed comparison across providers
2. **VRAM** — Actual nvidia-smi measurements
3. **Coding tasks** — Real-world task performance with pytest validation

## Setup

| Component     | Specification                        |
| ------------- | ------------------------------------ |
| **CPU**       | AMD Ryzen 7 5700G (16 threads)       |
| **RAM**       | 32 GB DDR4                           |
| **GPU**       | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS**        | Ubuntu 24.04.4 LTS                   |
| **Driver**    | NVIDIA 580.65.06                     |
| **llama.cpp** | b8149-6-g832aa9476                   |

## Models Tested

All models are IQ4-class quantizations (~18-20GB) of the same base model (Qwen3.5-35B-A3B MoE).

| ID                | Provider  | Quant     | Approach                                                                                                           |
| ----------------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------ |
| `aessedai-iq4xs`  | AesSedai  | IQ4_XS    | MoE-optimized: Q8_0 attention + IQ3_S FFN experts ([source](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF)) |
| `bartowski-iq4xs` | bartowski | IQ4_XS    | imatrix calibration ([source](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF))                         |
| `unsloth-mxfp4`   | unsloth   | MXFP4_MOE | MXFP4 format ([source](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF))                                       |

## Results (v2 — 2026-03-01)

### Summary Table

| Model            | Mean Score | 95% CI       | TPS  | Perfect Runs |
| ---------------- | ---------- | ------------ | ---- | ------------ |
| AesSedai IQ4_XS  | **85.7**   | [77.7, 93.7] | 27.3 | 1/15         |
| bartowski IQ4_XS | 84.7       | [78.4, 90.9] | 25.3 | 1/15         |
| unsloth MXFP4    | 83.7       | [75.8, 91.5] | **28.2** | 3/15     |

**Finding:** All three providers perform within margin of error on coding tasks. The confidence intervals overlap significantly, indicating no statistically significant difference.

### Score Distribution

| Model     | Min | Max | Std Dev | Scores                                                              |
| --------- | --- | --- | ------- | ------------------------------------------------------------------- |
| AesSedai  | 40  | 100 | 14.5    | 90, 90, 40, 90, 90, 90, 90, 90, 100, 90, 90, 90, 90, 90, 65         |
| bartowski | 65  | 100 | 11.3    | 65, 90, 90, 65, 90, 90, 100, 90, 90, 90, 65, 90, 90, 75, 90         |
| unsloth   | 65  | 100 | 14.2    | 65, 100, 65, 90, 90, 90, 100, 65, 100, 65, 90, 90, 90, 65, 90       |

### Level Pass Rates (15 iterations)

| Level | Task                           | Points | AesSedai | bartowski | unsloth |
| ----- | ------------------------------ | ------ | -------- | --------- | ------- |
| L1    | Basic queue (add/get, FIFO)    | 25     | 87%      | 80%       | 67%     |
| L2    | Retry with exponential backoff | 25     | 87%      | 87%       | 100%    |
| L3    | Priority scheduling            | 25     | 93%      | 100%      | 100%    |
| L4    | Find & fix concurrency bug     | 15     | 100%     | 100%      | 100%    |
| L5    | Multi-file refactoring         | 10     | 7%       | 20%       | 27%     |

**Observations:**
- L4 (concurrency bug fix) has 100% pass rate across all providers
- L5 (multi-file refactoring) is the hardest, with <30% pass rate
- L1 surprisingly has the most variance — syntax errors in code extraction

## Methodology Changes (v1 → v2)

| Aspect           | v1 (2026-02-27)                | v2 (2026-03-01)                       |
| ---------------- | ------------------------------ | ------------------------------------- |
| **Iterations**   | 5                              | 15 (better statistical significance)  |
| **Context size** | 40960 tokens                   | 40960 tokens (unchanged)              |
| **Validation**   | Lambda string matching         | PytestValidator (real pytest tests)   |
| **Sampler**      | temp=0.1, `/no_think` prefix   | temp=0.6, top_p=0.95 (THINKING_CODING)|
| **Prompts**      | ~50 chars each                 | 500+ chars with clear requirements    |
| **Code storage** | Best iteration only            | All iterations (full artifacts)       |

### v2 Test Details

**L1: Basic Queue** — Implement `add_job(fn, *args)` returning job_id, `get_result(job_id)` blocking until complete, FIFO ordering verified

**L2: Retry with Backoff** — Max 3 retries with exponential backoff (1s base), `JobFailedError` after exhaustion

**L3: Priority Queue** — `add_job(fn, *args, priority=0)`, higher priority executes first, same priority uses FIFO

**L4: Concurrency Bug Fix** — Given broken code with race condition in `self.results[job_id] = result`, fix with proper locking

**L5: Multi-file Refactor** — Split monolithic `queue.py` into `queue/{__init__,core,retry,priority}.py` maintaining all functionality

## Key Findings

1. **No significant difference between providers** — All three providers score within 2 points of each other (83.7-85.7), with overlapping confidence intervals.

2. **v2 methodology shows higher scores** — With proper prompts and thinking mode, all providers achieve 80%+ avg vs v1's erratic results.

3. **L5 remains challenging** — Multi-file refactoring has <30% success rate across all providers, suggesting this is a genuine capability limit.

4. **TPS varies slightly** — unsloth MXFP4 is marginally faster (28.2 TPS) but all are within 10% of each other.

## Recommendations

| Use Case              | Recommended                                 |
| --------------------- | ------------------------------------------- |
| **Tight VRAM budget** | AesSedai IQ4_XS (18.1 GB, lowest VRAM)      |
| **Maximum speed**     | unsloth MXFP4 (28.2 TPS, slightly faster)   |
| **Coding tasks**      | Any — no significant difference             |
| **Lowest variance**   | bartowski IQ4_XS (std dev 11.3)             |

## Files

| File                          | Description                              |
| ----------------------------- | ---------------------------------------- |
| `result_aessedai.json`        | Full benchmark results (15 iterations)   |
| `result_bartowski.json`       | Full benchmark results (15 iterations)   |
| `result_unsloth.json`         | Full benchmark results (15 iterations)   |
| `artifacts/*/iter_*/`         | Generated code per iteration per level   |

## References

- [Why Maybe We're Measuring LLM Compression Wrong](https://huggingface.co/blog/rishiraj/kld-guided-quantization) — KLD methodology
- [Qwen3.5-35B-A3B Q4 Quantization Comparison (Reddit)](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/) — Source of KLD/PPL data
- [AesSedai MoE-Optimized Explanation](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF) — Explains Q8_0 attention + lower FFN approach
