# Qwen3.5-35B-A3B Quantization Comparison

Benchmark comparing Q4_K_XL vs Q3_K_XL quantizations of the Qwen3.5-35B-A3B
MoE model (3B active parameters) for multi-agent workloads.

## Models Under Test

| Model | Preset | Port | VRAM | Context | File |
|-------|--------|------|------|---------|------|
| Qwen3-Coder-30B | (llama-swap) | 7070 | ~18 GB | 32k | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| Qwen3.5-35B Q4_K_XL | `qwen35-35b-multi` | 7080 | 22 GB | 32k | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` |
| Qwen3.5-35B Q3_K_XL | `qwen35-35b-q3-multi` | 7081 | 18 GB | 40k | `Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf` |

Qwen3.5 presets configured with `--parallel 3 --cont-batching --threads 16`.
Qwen3-Coder-30B data from existing benchmark (2026-02-22).

## Hypothesis

**Qwen3.5 vs Qwen3-Coder:**
- Qwen3.5-35B-A3B is a newer MoE model - expect better quality
- Both have 3B active parameters - similar speed expected

**Q4 vs Q3 quantization:**
- Q3_K_XL uses ~4GB less VRAM, allowing 8k more context (40k vs 32k)
- Trade-off: lower quantization may reduce quality

This benchmark measures:
1. **Quality comparison** across model generations and quant levels
2. **Throughput differences** under concurrent load
3. **TTFT stability** at various concurrency levels

## Prerequisites

Download the model files:

```bash
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --local-dir ~/bin

huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf --local-dir ~/bin
```

## Running the Benchmark

```bash
# Sync presets first
uv run gpumod preset sync

# Start Q4 model
uv run gpumod service start qwen35-35b-multi

# Run benchmark
uv run python docs/benchmarks/20260225_qwen35_quant_comparison/benchmark_quant.py \
  --model "Qwen3.5-35B Q4_K_XL" \
  --port 7080 \
  --output docs/benchmarks/20260225_qwen35_quant_comparison/

# Stop Q4, start Q3
uv run gpumod service stop qwen35-35b-multi
uv run gpumod service start qwen35-35b-q3-multi

# Run benchmark for Q3
uv run python docs/benchmarks/20260225_qwen35_quant_comparison/benchmark_quant.py \
  --model "Qwen3.5-35B Q3_K_XL" \
  --port 7081 \
  --output docs/benchmarks/20260225_qwen35_quant_comparison/
```

## Methodology

- **Temperature**: 0.2 (deterministic)
- **Workload**: Mixed coding tasks + concurrent side tasks
- **Concurrency levels**: 1, 2, 3 (max parallel slots)
- **Primary task**: Multi-turn FastAPI endpoint development
- **Side tasks**: Docstrings, unit tests, bug finding, async conversion

See [benchmark methodology](../README.md) for full scoring rubric.

## Expected Results

| Metric | Qwen3-Coder | Q4_K_XL | Q3_K_XL | Notes |
|--------|-------------|---------|---------|-------|
| Quality (1-5) | Baseline | Higher? | Slightly lower | Qwen3.5 is newer |
| TTFT @ C1 | 135ms | Similar | Similar | Same MoE arch |
| tok/s @ C3 | 29-82 | Similar | Slightly faster | Smaller weights |
| Side tasks | All PASS | TBD | TBD | Verify consistency |
| Max context | 32k | 32k | 40k | Q3 has headroom |

## Files

- `benchmark_quant.py` - Benchmark script
- `20260225_qwen3_coder_30b.json` - Qwen3-Coder-30B results (from 2026-02-22)
- `20260225_qwen3.5_35b_q4_k_xl.json` - Q4_K_XL results (to be generated)
- `20260225_qwen3.5_35b_q3_k_xl.json` - Q3_K_XL results (to be generated)
- `analysis.md` - Comparison analysis (generated after runs)
