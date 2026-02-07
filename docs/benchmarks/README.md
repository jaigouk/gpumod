# GPU Model Benchmark Framework

Reproducible benchmarking framework for comparing LLMs running on the
llama.cpp router (RTX 4090, 24 GB VRAM).

## Quick Start

```bash
# Run a single model benchmark
uv run python scripts/benchmark.py \
  --model nemotron-3-nano \
  --output docs/benchmarks/20260207_nemotron_devstral/

# Run lifecycle (mode-switching) test
uv run python scripts/benchmark.py \
  --lifecycle \
  --models nemotron-3-nano,devstral-small-2 \
  --output docs/benchmarks/20260207_nemotron_devstral/

# Generate comparison charts from scored JSON files
uv run python scripts/benchmark_chart.py \
  docs/benchmarks/20260207_nemotron_devstral/*.json \
  --output docs/benchmarks/20260207_nemotron_devstral/charts/
```

## Directory Layout

```
docs/benchmarks/
  README.md                          <- this file (methodology)
  20260207_nemotron_devstral/        <- one folder per benchmark run
    20260207_nemotron.json           <- raw results + scores
    20260207_devstral.json
    20260207_lifecycle.json
    20260207_nemotron_devstral_report.md
    charts/
      radar.png
      performance.png
      lifecycle.png
```

## Methodology

### Temperature

All models are tested with **temperature = 0.2** for deterministic,
comparable outputs. This overrides any model-specific preset defaults.

### Prompt Categories (Radar Chart Axes)

| # | Category | Prompts | Hallucination Detection |
|---|----------|---------|------------------------|
| 1 | Factual Knowledge | 3 | Yes - ground truth verification |
| 2 | Reasoning and Logic | 3 | No |
| 3 | Code Generation | 3 | No |
| 4 | Tool Use / Structured Output | 2 | No |
| 5 | Writing and Summarization | 2 | No |

### Hallucination Detection (Factual Knowledge)

Each factual prompt has a **ground truth** consisting of verifiable facts.
After the model responds, the checker runs two tests:

1. **Presence check** - required keywords that MUST appear in the response.
   Missing = incomplete answer (score penalty, not hallucination).
2. **Forbidden-claim check** - keywords indicating a known false claim.
   Present = hallucination (e.g., Pluto listed as a planet).

Metrics produced per factual prompt:

| Metric | Meaning |
|--------|---------|
| `facts_correct` | Verified facts found in response |
| `facts_missing` | Expected facts not mentioned |
| `facts_hallucinated` | Forbidden/false claims detected |
| `hallucination_rate` | hallucinated / (correct + hallucinated) |

### Performance Metrics

| Metric | Source | Unit |
|--------|--------|------|
| Time to First Token (TTFT) | Client-side streaming | ms |
| Generation speed | Server `timings` | tok/s |
| Prompt processing speed | Server `timings` | tok/s |
| Total response time | Client-side | ms |
| VRAM (peak) | `nvidia-smi` | MB |
| Model load time | Router load API | ms |
| Model unload time | Router unload API | ms |

### Mode-Switching Lifecycle Test

Measures user-visible latency for mode transitions:

```
blank -> code_model -> RAG (embedding) -> code_model
```

Each step:
1. **Unload** current model (timed)
2. **Load** next model (timed)
3. **Verification request** - simple prompt confirming model works (timed)

Total switch time = unload + load + first-request TTFT.

### Quality Scoring

All responses are scored **1-5** by the evaluator (Claude Opus 4.6) after
the benchmark run completes. Scoring rubric:

| Score | Meaning |
|-------|---------|
| 5 | Perfect - complete, accurate, well-formatted |
| 4 | Good - minor issues, mostly correct |
| 3 | Acceptable - some errors or missing info |
| 2 | Poor - significant errors or incomplete |
| 1 | Failed - wrong answer or gibberish |

For factual prompts, the hallucination rate adjusts the score:
- 0% hallucination + all facts present -> 5
- 0% hallucination + some facts missing -> 4
- Any hallucination detected -> capped at 2

## Adding a New Model

1. Register the model in `llama_cpp_gguf_presets.ini`
2. Run the benchmark script with `--model <model-id>`
3. The evaluator scores the responses
4. Regenerate charts
