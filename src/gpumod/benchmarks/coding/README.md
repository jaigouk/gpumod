# Coding Benchmark Suite

Benchmark suite for evaluating chat-completion models on coding tasks (5 levels, pytest-validated).

Originally built for Qwen 3.5 (hence the historical examples and naming below); now used across Qwen 3.5 / Qwen 3.6 / Gemma 4 with per-model sampler config and the same v2 methodology.

## Background

This suite was created based on community feedback on the original provider comparison benchmark, which identified several methodology issues:

- **Sample size too small** — Only 5 levels, effectively 3 discriminating
- **Wrong sampler settings** — Used temp=0.1 (near-greedy), Qwen recommends temp=0.6-0.7
- **Context size insufficient** — Used 8k, Qwen supports up to 262k
- **Metrics misleading** — "Best of N" inflates differences vs mean

## Verified Settings

All settings verified from official sources (February 2026):

### Sampler Configuration

From [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B):

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|------|-------|-------|-------|------------------|
| **Thinking (coding)** | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

**Warning**: Do NOT use greedy decoding (temp=0) — it breaks reasoning models.

### Server Configuration

From [llama.cpp MoE Guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) and [KV cache testing](https://github.com/ggml-org/llama.cpp/issues/11200):

| Flag | Value | Purpose |
|------|-------|---------|
| `-ctk` | q8_0 | KV cache key quantization (+50-100% speed) |
| `-ctv` | q8_0 | KV cache value quantization |
| `-fa` | on | Flash attention (now default) |
| `--fit` | on | Auto VRAM management for MoE |
| `--no-mmap` | — | Disable memory mapping |
| `--jinja` | — | Use embedded chat template |
| `-c` | 32768+ | Context size (32k minimum) |

### Benchmark Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Iterations | 15 | Reduces variance, enables 95% CI |
| Context | 32k+ | Qwen native is 262k |
| Primary metric | Mean | More stable than "best of N" |
| Secondary metrics | Std, 95% CI, min, max | Full statistical picture |

## Module Structure

```
src/gpumod/benchmarks/coding/
├── README.md              # This file
├── __init__.py
├── sampler_config.py      # THINKING_CODING, NON_THINKING presets
├── server_config.py       # DEFAULT_24GB, DEFAULT_16GB presets
├── prompt_categories.py   # SHORT, MEDIUM, LONG, MULTI_TURN
├── performance_metrics.py # TPS, VRAM, latency measurement
├── scoring.py             # Stats, confidence intervals
├── runner.py              # Benchmark orchestration
└── levels/                # Graduated difficulty test cases
    └── __init__.py
```

## Usage

```python
from gpumod.benchmarks.coding.sampler_config import THINKING_CODING
from gpumod.benchmarks.coding.server_config import DEFAULT_24GB
from gpumod.benchmarks.coding.runner import BenchmarkConfig, BenchmarkRunner

# Configure benchmark
config = BenchmarkConfig(
    model_id="unsloth/Qwen3.5-35B-A3B-GGUF",
    iterations=15,
    context_size=32768,
    sampler=THINKING_CODING,
)

# Run benchmark
runner = BenchmarkRunner(config)
runner.set_client(your_llm_client)
results = await runner.run()

# Generate report
report = runner.generate_report()
print(f"Mean: {report['stats']['mean']:.1f}")
print(f"95% CI: [{report['confidence_interval']['lower']:.1f}, {report['confidence_interval']['upper']:.1f}]")
```

## Prompt Categories

| Category | Target Tokens | Use Case |
|----------|---------------|----------|
| SHORT | ~100 | Quick function implementation |
| MEDIUM | ~500 | Class with multiple methods |
| LONG | ~2000 | Full system design |
| MULTI_TURN | ~1000 (cumulative) | Conversation with follow-ups |

## Sources

- [Qwen3.5-35B-A3B Model Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen llama.cpp Documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Unsloth Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [llama.cpp MoE Offload Guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [KV Cache Quantization Testing](https://github.com/ggml-org/llama.cpp/issues/11200)
- [Benchmark2 Paper](https://arxiv.org/pdf/2601.03986) (bootstrap CI methodology)
