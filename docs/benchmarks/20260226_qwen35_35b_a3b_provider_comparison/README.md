# Qwen3.5-35B-A3B IQ4 Provider Comparison (KLD Validation)

**Date:** 2026-02-27
**Ticket:** gpumod-56a

## Goal

Test whether KLD (faithfulness metric) predicts real-world coding task performance, and measure speed/VRAM tradeoffs across GGUF providers.

## Background

KLD/PPL rankings for Qwen3.5-35B-A3B quantizations already exist ([Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/)). This benchmark adds:

1. **TPS** — Speed comparison across providers
2. **VRAM** — Actual nvidia-smi measurements (not file size)
3. **Coding tasks** — Validate if KLD predicts task performance

## Setup

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 7 5700G (16 threads) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS** | Ubuntu 24.04.4 LTS |
| **Driver** | NVIDIA 580.65.06 |
| **llama.cpp** | b8149-6-g832aa9476 |

## Models Tested

All models are IQ4-class quantizations (~18-20GB) of the same base model (Qwen3.5-35B-A3B MoE).

| ID | Provider | Quant | Approach |
|----|----------|-------|----------|
| `aessedai-iq4xs` | AesSedai | IQ4_XS | MoE-optimized: Q8_0 attention + IQ3_S/IQ4_XS FFN experts |
| `bartowski-iq4xs` | bartowski | IQ4_XS | Standard imatrix quantization |
| `unsloth-mxfp4` | unsloth | MXFP4_MOE | MXFP4 format (post-hoc, not QAT) |

## Hypotheses

**H1:** Lower KLD → better coding performance
**H2:** MoE-optimized quants (Q8_0 attention) are slower
**H3:** Lower KLD → lower variance in coding tasks

## Results

### Summary Table

| Model | KLD | PPL | TPS | VRAM (GB) | Coding Best | Coding Avg | Coding Std |
|-------|-----|-----|-----|-----------|-------------|------------|------------|
| AesSedai IQ4_XS | **0.0240** | 6.517 | 123.28 ± 0.89 | **18.1** | 50 | 33.0 | 19.2 |
| bartowski IQ4_XS | 0.0243 | **6.512** | **135.81** ± 3.02 | 19.4 | 25 | 10.0 | 12.2 |
| unsloth MXFP4 | 0.0253 | 6.485 | 123.18 ± 0.49 | 20.2 | **90** | **55.0** | 20.6 |

*KLD/PPL data from [Qwen3.5-35B-A3B Q4 Quantization Comparison](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/)*

### TPS Benchmark (3 runs each)

| Model | Mean TPS | Std Dev | VRAM Idle | VRAM Peak |
|-------|----------|---------|-----------|-----------|
| AesSedai IQ4_XS | 123.28 | 0.89 | 18119 MB | 18133 MB |
| bartowski IQ4_XS | 135.81 | 3.02 | 19419 MB | 19433 MB |
| unsloth MXFP4 | 123.18 | 0.49 | 20199 MB | 20213 MB |

**Finding:** bartowski is ~10% faster (135 vs 123 TPS) but uses ~1.3GB more VRAM than AesSedai.

### Job Queue Challenge (5 iterations each)

| Model | L1 | L2 | L3 | L4 | L5 | Best | Avg | All Scores |
|-------|----|----|----|----|-----|------|-----|------------|
| AesSedai IQ4_XS | 25 | 0 | 25 | 0 | 0 | 50 | 33.0 | [50, 0, 40, 25, 50] |
| bartowski IQ4_XS | 25 | 0 | 0 | 0 | 0 | 25 | 10.0 | [0, 0, 25, 0, 25] |
| unsloth MXFP4 | 25 | 25 | 25 | 15 | 0 | **90** | **55.0** | [90, 40, 65, 40, 40] |

**Finding:** unsloth MXFP4 achieved 90/100 (only missing L5 refactoring), significantly outperforming models with better KLD.

## Hypothesis Analysis

### H1: Lower KLD → better coding performance

**FALSIFIED**

| Rank by KLD | Model | KLD | Coding Best |
|-------------|-------|-----|-------------|
| 1 (best) | AesSedai | 0.0240 | 50 |
| 2 | bartowski | 0.0243 | 25 |
| 3 (worst) | unsloth | 0.0253 | **90** |

The model with the **worst KLD** (unsloth, 0.0253) achieved the **best coding score** (90). The relationship is inverted from expectations.

**Interpretation:** KLD measures faithfulness to BF16 baseline, but for IQ4-class quantizations where KLD differences are small (0.024-0.025), this doesn't translate to practical task performance. Other factors dominate:
- Format-specific optimizations (MXFP4 may handle certain operations better)
- Model response reliability (all models had "empty response" issues)

### H2: MoE-optimized quants (Q8_0 attention) are slower

**PARTIALLY SUPPORTED**

| Model | Q8_0 Attention? | TPS |
|-------|-----------------|-----|
| AesSedai | Yes | 123.28 |
| bartowski | No | **135.81** |
| unsloth | No | 123.18 |

AesSedai (with Q8_0 attention) is ~10% slower than bartowski. However, unsloth is equally slow without Q8_0 attention, suggesting other factors (MXFP4 format overhead?) also affect speed.

### H3: Lower KLD → lower variance

**INCONCLUSIVE**

| Model | KLD | Coding Std Dev |
|-------|-----|----------------|
| AesSedai | 0.0240 | 19.2 |
| bartowski | 0.0243 | **12.2** |
| unsloth | 0.0253 | 20.6 |

No clear pattern. All models exhibit high variance due to inconsistent response generation (frequent empty responses).

## Key Findings

1. **KLD is not predictive of coding performance** at IQ4-class quantization levels. The 0.001-0.002 KLD differences between providers don't translate to measurable task quality differences.

2. **unsloth MXFP4 outperformed expectations** — Despite having the worst KLD (0.0253), it achieved the best coding score (90/100). This suggests MXFP4 format may have advantages for certain task types that aren't captured by KLD.

3. **Speed/VRAM tradeoff exists** — bartowski is 10% faster but uses 7% more VRAM than AesSedai. Choose based on your constraint (speed vs memory).

4. **All models have reliability issues** — Empty responses occurred across all providers, causing high variance. This is likely a Qwen3.5 thinking model behavior issue, not provider-specific.

## Recommendations

| Use Case | Recommended |
|----------|-------------|
| **Tight VRAM budget** | AesSedai IQ4_XS (18.1 GB, best KLD) |
| **Maximum speed** | bartowski IQ4_XS (136 TPS, 10% faster) |
| **Best coding tasks** | unsloth MXFP4 (90/100 best score) |
| **Balanced** | AesSedai or unsloth depending on task type |

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

| File | Description |
|------|-------------|
| `tps_*.json` | TPS benchmark results |
| `result_*.json` | Coding benchmark results |
| `artifacts/*/` | Generated code from best iteration |
| `benchmark_runner.py` | Coding benchmark script |
| `tps_benchmark.py` | TPS measurement script |
| `test_job_queue.py` | pytest test suite |

## References

- [Why Maybe We're Measuring LLM Compression Wrong](https://huggingface.co/blog/rishiraj/kld-guided-quantization) — KLD methodology, why it's better than PPL
- [Qwen3.5-35B-A3B Q4 Quantization Comparison (Reddit)](https://www.reddit.com/r/LocalLLaMA/comments/1rfds1h/qwen3535ba3b_q4_quantization_comparison/) — Source of KLD/PPL data used in this benchmark
- [AesSedai MoE-Optimized Explanation](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF) — Explains Q8_0 attention + lower FFN approach
