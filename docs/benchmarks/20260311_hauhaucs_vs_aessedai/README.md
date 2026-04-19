# Qwen3.5-35B-A3B IQ4_XS: HauhauCS Uncensored vs AesSedai

**Date:** 2026-03-11

## Goal

Compare the **HauhauCS Uncensored** fine-tune against the **AesSedai** quantization on coding tasks, speed, and VRAM.

## Background

- **AesSedai IQ4_XS**: MoE-optimized quant with Q8_0 attention + IQ3_S FFN experts
- **HauhauCS IQ4_XS**: Uncensored fine-tune with "aggressive" personality adjustments

Both use the same base model (Qwen3.5-35B-A3B MoE) and IQ4_XS quantization, but differ in:
1. **Fine-tuning**: HauhauCS applies uncensoring and personality changes
2. **Quantization approach**: Different calibration/imatrix data

## Setup

| Component     | Specification                        |
| ------------- | ------------------------------------ |
| **CPU**       | AMD Ryzen 7 5700G (16 threads)       |
| **RAM**       | 32 GB DDR4                           |
| **GPU**       | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| **OS**        | Ubuntu 24.04.4 LTS                   |
| **Driver**    | NVIDIA 580.65.06                     |
| **llama.cpp** | TBD                                  |
| **context_size** | 40960 tokens                      |

## Models Tested

| ID           | Provider  | Quant  | Source                                                                                                               |
| ------------ | --------- | ------ | -------------------------------------------------------------------------------------------------------------------- |
| `hauhaucs`   | HauhauCS  | IQ4_XS | [HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive) |
| `aessedai`   | AesSedai  | IQ4_XS | [AesSedai/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/AesSedai/Qwen3.5-35B-A3B-GGUF)                                |

## Methodology

Uses the **Job Queue Challenge** benchmark (v2 methodology from 20260226_qwen35_35b_a3b_provider_comparison):

- **15 iterations** per model for statistical significance
- **5 levels** of increasing difficulty (L1-L5)
- **pytest validation** for each level
- **Sampler**: temp=0.6, top_p=0.95, top_k=20

### Levels

| Level | Task                           | Points |
| ----- | ------------------------------ | ------ |
| L1    | Basic queue (add/get, FIFO)    | 25     |
| L2    | Retry with exponential backoff | 25     |
| L3    | Priority scheduling            | 25     |
| L4    | Find & fix concurrency bug     | 15     |
| L5    | Multi-file refactoring         | 10     |

## Results

### Summary Table

| Model            | Mean Score | 95% CI | TPS  | VRAM (MB) |
| ---------------- | ---------- | ------ | ---- | --------- |
| HauhauCS IQ4_XS  | TBD        | TBD    | TBD  | TBD       |
| AesSedai IQ4_XS  | TBD        | TBD    | TBD  | TBD       |

### Score Distribution

| Model     | Min | Max | Std Dev | Scores |
| --------- | --- | --- | ------- | ------ |
| HauhauCS  | TBD | TBD | TBD     | TBD    |
| AesSedai  | TBD | TBD | TBD     | TBD    |

## Running the Benchmark

```bash
# Start HauhauCS service
gpumod start qwen35-35b-a3b-hauhaucs-iq4xs

# Run benchmark (15 iterations)
uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
    --model hauhaucs --port 7097 --iterations 15 \
    --output docs/benchmarks/20260311_hauhaucs_vs_aessedai/

# Stop and switch
gpumod stop qwen35-35b-a3b-hauhaucs-iq4xs
gpumod start qwen35-35b-a3b-aessedai-iq4xs

# Run benchmark for AesSedai
uv run python docs/benchmarks/20260226_qwen35_job_queue_challenge/benchmark_runner.py \
    --model aessedai --port 7094 --iterations 15 \
    --output docs/benchmarks/20260311_hauhaucs_vs_aessedai/
```

## Files

| File                   | Description                     |
| ---------------------- | ------------------------------- |
| `result_hauhaucs.json` | HauhauCS benchmark results      |
| `result_aessedai.json` | AesSedai benchmark results      |
| `artifacts/*/iter_*/`  | Generated code per iteration    |

## Key Questions

1. Does the HauhauCS fine-tune affect coding capability?
2. Is there a speed difference from the uncensoring process?
3. Do the "aggressive" personality changes impact code quality?
