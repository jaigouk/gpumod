# Qwen3.6 vs Gemma4 Architecture Comparison

**Date:** 2026-04-24
**Ticket:** gpumod-4la

## Goal

Compare three local LLMs with different architectures on coding task performance, TPS, and VRAM usage on RTX 4090 (24GB VRAM).

## Setup

| Component     | Specification                        |
| ------------- | ------------------------------------ |
| **CPU**       | AMD Ryzen 7 5700G (16 threads)       |
| **RAM**       | 32 GB DDR4                           |
| **GPU**       | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS**        | Ubuntu 24.04.4 LTS                   |
| **Driver**    | NVIDIA 580.65.06                     |
| **llama.cpp** | b8838 (23b8cc499)                    |

## Models Tested

| ID               | Model            | Architecture              | Quant    | File Size | VRAM est. |
| ---------------- | ---------------- | ------------------------- | -------- | --------- | --------- |
| `qwen36-27b`     | Qwen3.6-27B      | Dense (27B all active)    | Q4_K_M   | 16.0 GB   | ~18 GB    |
| `qwen36-35b-a3b` | Qwen3.6-35B-A3B  | MoE (35B total, 3B active)| UD-Q4_K_S| 19.9 GB   | ~22 GB    |
| `gemma4-e4b`     | Gemma 4 E4B      | Dense (full precision)    | BF16     | 15.0 GB   | ~16 GB    |

## Results

### Summary Table

| Model                         | Architecture    | Quant      | Mean Score | Std Dev | 95% CI       | TPS       | Perfect Runs |
| ----------------------------- | --------------- | ---------- | ---------- | ------- | ------------ | --------- | ------------ |
| Qwen3.6-35B-A3B               | MoE (3B active) | UD-Q4_K_S  | **90.0**   | **0.0** | [90.0, 90.0] | **173.7** | 0/15         |
| Gemma 4 E4B                   | Dense           | BF16       | 88.3       | 6.5     | [84.8, 91.9] | 82.9      | 0/15         |
| Qwen3.5-35B-A3B (AesSedai)†   | MoE (3B active) | IQ4_XS     | 85.7       | 14.5    | [77.7, 93.7] | 27.3†     | 1/15         |
| Qwen3.5-35B-A3B (bartowski)†  | MoE (3B active) | IQ4_XS     | 84.7       | 11.3    | [78.4, 90.9] | 25.3†     | 1/15         |
| Qwen3.5-35B-A3B (unsloth)†    | MoE (3B active) | MXFP4      | 83.7       | 14.2    | [75.8, 91.5] | 28.2†     | 3/15         |
| Qwen3.6-27B                   | Dense (27B)     | Q4_K_M     | 80.3       | 6.9     | [76.5, 84.2] | 46.9      | 0/15         |

**95% CI** (Confidence Interval): the range where the true mean score likely falls 95% of the time. A narrow CI like [90.0, 90.0] means highly consistent results; a wide CI like [75.8, 91.5] means high variance across runs. When CIs overlap between models, the difference is not statistically significant.

**Perfect Runs**: iterations that scored 100/100 (all 5 levels passed). No model in this benchmark achieved a perfect run because L5 (multi-file refactoring) was never solved. The Qwen3.5 models occasionally scored 100 in the prior benchmark due to different L5 behavior.

† Qwen3.5 results from [prior benchmark (2026-02-27)](../20260226_qwen35_35b_a3b_provider_comparison/README.md), same v2 methodology. TPS measured via `X-Llama-Timings` header (may undercount thinking tokens).

### Score Distribution

| Model            | Min | Max | Scores                                                            |
| ---------------- | --- | --- | ----------------------------------------------------------------- |
| Qwen3.6-35B-A3B  | 90  | 90  | 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90       |
| Gemma 4 E4B      | 65  | 90  | 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 65, 90       |
| Qwen3.6-27B      | 75  | 90  | 90, 90, 85, 75, 75, 75, 75, 75, 90, 75, 85, 75, 75, 90, 75       |

### Level Pass Rates (15 iterations)

Each iteration runs the model through 5 coding tasks of increasing difficulty. The model receives a prompt, generates Python code, and the output is validated by running real pytest tests against it. A level passes only if all tests pass.

- **L1 (25 pts):** Implement a basic job queue with `add_job()` and `get_result()`, verified by FIFO ordering tests.
- **L2 (25 pts):** Add retry logic with exponential backoff (1s base, max 3 retries), raise `JobFailedError` after exhaustion.
- **L3 (25 pts):** Implement priority scheduling — higher priority jobs execute first, same priority uses FIFO.
- **L4 (15 pts):** Given broken code with a race condition in `self.results[job_id] = result`, diagnose and fix with proper locking.
- **L5 (10 pts):** Split a monolithic `queue.py` into `queue/{__init__,core,retry,priority}.py` while maintaining all functionality.

| Level | Task                           | Points | Qwen3.6-27B | Qwen3.6-35B-A3B | Gemma 4 E4B |
| ----- | ------------------------------ | ------ | ----------- | ---------------- | ----------- |
| L1    | Basic queue (add/get, FIFO)    | 25     | 100%        | 100%             | 100%        |
| L2    | Retry with exponential backoff | 25     | 100%        | 100%             | 93%         |
| L3    | Priority scheduling            | 25     | 100%        | 100%             | 100%        |
| L4    | Find & fix concurrency bug     | 15     | 27%         | **100%**         | **100%**    |
| L5    | Multi-file refactoring         | 10     | 13%         | 0%               | 0%          |

## Key Findings

1. **MoE dominates dense on every metric.** Qwen3.6-35B-A3B scored 90/100 on every single iteration — zero variance across 15 runs. It is also 3.7x faster than the 27B dense model (173.7 vs 46.9 TPS) because only 3B of 35B params are active per token.

2. **Gemma 4 E4B punches above its weight.** A smaller model at full BF16 precision scores 88.3 mean — within striking distance of the 35B-A3B MoE — at 82.9 TPS. One outlier iteration (65) dragged the mean down.

3. **Dense 27B struggles with L4 (concurrency).** Only 27% pass rate on the bug-fix task. With thinking mode enabled, the model generates extensive reasoning tokens, sometimes timing out or producing incomplete code. The 35B-A3B and Gemma4 handle it reliably (100%).

4. **L5 (multi-file refactoring) remains unsolved.** 0% for 35B-A3B and Gemma4, 13% for 27B. This confirms the finding from the prior Qwen3.5 benchmark — multi-file refactoring is a genuine capability ceiling for local models at this scale.

5. **TPS correlates with active parameter count**, not total model size:
   - 3B active (35B-A3B MoE): 173.7 TPS
   - ~4B (Gemma4 E4B): 82.9 TPS
   - 27B active (dense): 46.9 TPS

## Methodology

Reuses v2 methodology from the [Qwen3.5 provider comparison](../20260226_qwen35_35b_a3b_provider_comparison/README.md):

| Aspect           | Configuration                         |
| ---------------- | ------------------------------------- |
| **Iterations**   | 15                                    |
| **Validation**   | PytestValidator (real pytest tests)   |
| **Sampler**      | temp=0.6, top_p=0.95 (THINKING_CODING)|
| **Timeout**      | 300s per request                      |
| **Context size** | 40960 (27B), 32768 (35B-A3B), 65536 (Gemma4) |

## Recommendations

| Use Case              | Recommended                                     |
| --------------------- | ----------------------------------------------- |
| **Best overall**      | Qwen3.6-35B-A3B — highest quality, fastest      |
| **Lowest VRAM**       | Gemma 4 E4B (~16 GB, near-equal quality)         |
| **Maximum speed**     | Qwen3.6-35B-A3B (173.7 TPS)                     |
| **Lowest variance**   | Qwen3.6-35B-A3B (std dev 0.0)                   |
| **Budget GPU (<20GB)**| Gemma 4 E4B — best quality under 20 GB VRAM     |

## Files

| File                          | Description                              |
| ----------------------------- | ---------------------------------------- |
| `result_qwen36-27b.json`      | Full benchmark results (15 iterations)   |
| `result_qwen36-35b-a3b.json`  | Full benchmark results (15 iterations)   |
| `result_gemma4-e4b.json`      | Full benchmark results (15 iterations)   |
| `artifacts/*/iter_*/`         | Generated code per iteration per level   |

## References

- [Prior benchmark: Qwen3.5-35B-A3B Provider Comparison](../20260226_qwen35_35b_a3b_provider_comparison/README.md)
- [Qwen3.6 Blog Post](https://qwen.ai/blog?id=qwen3.6-27b)
- [Unsloth Qwen3.6 Docs](https://unsloth.ai/docs/models/qwen3.6)
