# Qwen3.5-35B-A3B IQ4 Provider Comparison

**Date:** 2026-02-27

## Goal

Compare IQ4-class GGUF quantizations from different providers on speed, VRAM, and coding task performance. Explore whether reported KLD values correlate with task performance.

## Background

KLD/PPL rankings for Qwen3.5-35B-A3B quantizations already exist ([Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/)). This benchmark adds:

1. **TPS (tokens per second)** — Speed comparison across providers
2. **VRAM** — Actual nvidia-smi measurements (not file size)
3. **Coding tasks** — Real-world task performance comparison

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

## Questions Explored

- **Q1:** Does lower KLD correlate with better coding performance?
- **Q2:** Are MoE-optimized quants (Q8_0 attention) slower?
- **Q3:** Does lower KLD correlate with lower variance in coding tasks?

## Results

### Summary Table

| Model            | KLD        | PPL       | TPS               | VRAM (GB) | Coding Best | Coding Avg | Coding Std |
| ---------------- | ---------- | --------- | ----------------- | --------- | ----------- | ---------- | ---------- |
| AesSedai IQ4_XS  | **0.0240** | 6.517     | 123.28 ± 0.89     | **18.1**  | 50          | 33.0       | 19.2       |
| bartowski IQ4_XS | 0.0243     | **6.512** | **135.81** ± 3.02 | 19.4      | 25          | 10.0       | 12.2       |
| unsloth MXFP4    | 0.0253     | 6.485     | 123.18 ± 0.49     | 20.2      | **90**      | **55.0**   | 20.6       |

_KLD/PPL are reported values from [Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/). HuggingFace model cards may show different values due to test conditions._

### TPS Benchmark (3 runs each)

| Model            | Mean TPS | Std Dev | VRAM Idle | VRAM Peak |
| ---------------- | -------- | ------- | --------- | --------- |
| AesSedai IQ4_XS  | 123.28   | 0.89    | 18119 MB  | 18133 MB  |
| bartowski IQ4_XS | 135.81   | 3.02    | 19419 MB  | 19433 MB  |
| unsloth MXFP4    | 123.18   | 0.49    | 20199 MB  | 20213 MB  |

**Finding:** bartowski is ~10% faster (135 vs 123 TPS) but uses ~1.3GB more VRAM than AesSedai.

### Job Queue Challenge (5 iterations each)

| Model            | L1  | L2  | L3  | L4  | L5  | Best   | Avg      | All Scores           |
| ---------------- | --- | --- | --- | --- | --- | ------ | -------- | -------------------- |
| AesSedai IQ4_XS  | 25  | 0   | 25  | 0   | 0   | 50     | 33.0     | [50, 0, 40, 25, 50]  |
| bartowski IQ4_XS | 25  | 0   | 0   | 0   | 0   | 25     | 10.0     | [0, 0, 25, 0, 25]    |
| unsloth MXFP4    | 25  | 25  | 25  | 15  | 0   | **90** | **55.0** | [90, 40, 65, 40, 40] |

#### Difficulty Levels

| Level | Task                           | Points | Pass Rate   |
| ----- | ------------------------------ | ------ | ----------- |
| L1    | Basic queue (add/get, FIFO)    | 25     | 100% (3/3)  |
| L2    | Retry with exponential backoff | 25     | 33% (1/3)   |
| L3    | Priority scheduling            | 25     | 67% (2/3)   |
| L4    | Find & fix concurrency bug     | 15     | 33% (1/3)   |
| L5    | Multi-file refactoring         | 10     | 0% (0/3)    |

*Pass rate = models that passed at least once in 5 iterations*

#### Score Interpretation

| Score   | Levels Passed        | Interpretation                         |
| ------- | -------------------- | -------------------------------------- |
| 0–25    | L1 only              | Basic: Fundamental queue operations    |
| 26–50   | L1 + (L2 or L3)      | Intermediate: Retry OR priority        |
| 51–75   | L1 + L2 + L3         | Advanced: Both retry and priority      |
| 76–90   | L1 + L2 + L3 + L4    | Expert: Includes concurrency bug fix   |
| 91–100  | All levels           | Complete: Full multi-file refactoring  |

#### Test Details

**L1: Basic Queue** — `add_job()` returns job_id, `get_result()` returns value, FIFO ordering

**L2: Retry with Backoff** — Max 3 retries with exponential backoff (1s, 2s, 4s)

**L3: Priority Queue** — Higher priority executes first, same priority uses FIFO

**L4: Concurrency Bug Fix** — Find race condition in `self.results[job_id] = result`, fix with proper locking

**L5: Multi-file Refactor** — Split monolithic `queue.py` into `queue/{__init__,core,retry,priority}.py`

**Finding:** unsloth MXFP4 achieved 90/100 (only missing L5 refactoring). See [Job Queue Challenge](../20260226_qwen35_job_queue_challenge/README.md) for full benchmark details.

## Observations

### Q1: KLD vs Coding Performance

| Rank by KLD | Model     | KLD    | Coding Best |
| ----------- | --------- | ------ | ----------- |
| 1 (lowest)  | AesSedai  | 0.0240 | 50          |
| 2           | bartowski | 0.0243 | 25          |
| 3 (highest) | unsloth   | 0.0253 | **90**      |

The model with the highest KLD (unsloth, 0.0253) achieved the best coding score (90). The model with the lowest KLD (AesSedai, 0.0240) scored 50. No correlation observed in this test. Sample size: 3 models, 5 iterations each.

### Q2: Quantization Strategy vs Speed

| Model     | Quantization Strategy      | TPS        |
| --------- | -------------------------- | ---------- |
| AesSedai  | Q8_0 attention + IQ3_S FFN | 123.28     |
| bartowski | imatrix calibration        | **135.81** |
| unsloth   | MXFP4 format               | 123.18     |

bartowski is ~10% faster than both AesSedai and unsloth. The cause is not determined—multiple factors affect TPS.

### Q3: KLD vs Variance

| Model     | KLD    | Coding Std Dev |
| --------- | ------ | -------------- |
| AesSedai  | 0.0240 | 19.2           |
| bartowski | 0.0243 | **12.2**       |
| unsloth   | 0.0253 | 20.6           |

No clear pattern. All models exhibit high variance (frequent empty responses across all providers).

## Key Findings

1. **KLD did not correlate with coding performance** in this test. The model with highest KLD (unsloth, 0.0253) achieved the best coding score (90/100). Sample size is limited (3 models, 5 iterations).

2. **unsloth MXFP4 scored highest** on the coding benchmark (90/100 best, 55.0 avg) despite having the highest KLD.

3. **Speed/VRAM tradeoff exists** — bartowski is 10% faster (135 vs 123 TPS) but uses 7% more VRAM (19.4 vs 18.1 GB) than AesSedai.

4. **High variance across all models** — Empty responses occurred across all providers, causing inconsistent scores between iterations.

## Recommendations

| Use Case              | Recommended                                |
| --------------------- | ------------------------------------------ |
| **Tight VRAM budget** | AesSedai IQ4_XS (18.1 GB, best KLD)        |
| **Maximum speed**     | bartowski IQ4_XS (135 TPS, 10% faster)     |
| **Best coding tasks** | unsloth MXFP4 (90/100 best score)          |
| **Balanced**          | AesSedai or unsloth depending on task type |

## Methodology

### TPS Measurement

- Standardized prompt: ~850 tokens input (code analysis task)
- Output: 200 tokens max
- 3 runs per model, warmup run excluded
- Settings: `--parallel 1 --threads 16 -ngl -1`

### VRAM Measurement

- nvidia-smi during idle (model loaded) and peak (generation)
- Context: 40960 tokens configured

### Coding Benchmark

- Job Queue Challenge: 5 levels (L1-L5), graduated difficulty
- 5 iterations per model to capture variance
- Settings: `temperature=0.1, max_tokens=8192`
- `/no_think` prefix to skip reasoning phase

## Files

| File                  | Description                        |
| --------------------- | ---------------------------------- |
| `tps_*.json`          | TPS benchmark results              |
| `result_*.json`       | Coding benchmark results           |
| `artifacts/*/`        | Generated code from best iteration |
| `benchmark_runner.py` | Coding benchmark script            |
| `tps_benchmark.py`    | TPS measurement script             |
| `test_job_queue.py`   | pytest test suite                  |

## References

- [Why Maybe We're Measuring LLM Compression Wrong](https://huggingface.co/blog/rishiraj/kld-guided-quantization) — KLD methodology, why it's better than PPL
- [Qwen3.5-35B-A3B Q4 Quantization Comparison (Reddit)](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/) — Source of KLD/PPL data used in this benchmark
- [AesSedai MoE-Optimized Explanation](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF) — Explains Q8_0 attention + lower FFN approach
