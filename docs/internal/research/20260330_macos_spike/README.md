# macOS Apple Silicon — Phase 1 Spike Verification Scripts

**Spike ticket:** gpumod-2qc
**Epic:** gpumod-dwm

Runnable scripts that answer the spike's 16 research questions with measured data.
All output goes to `*_results.json` files in this directory.

## Quick Start

```bash
# Day 1: GPU metrics (Q1-Q4) — runs on this machine right now
uv run python docs/internal/research/20260330_macos_spike/verify_gpu_metrics.py

# Day 1: RAM portability (Q11) — runs on this machine right now
uv run python docs/internal/research/20260330_macos_spike/verify_ram_portability.py

# Day 2: launchd lifecycle (Q7-Q9) — creates/removes temp plists
uv run python docs/internal/research/20260330_macos_spike/verify_launchd.py

# Day 2: Logging and Metal patterns (Q10, Q16)
uv run python docs/internal/research/20260330_macos_spike/verify_logging.py

# Day 1-2: vLLM endpoint probe (Q5) — requires running server
# Terminal 1: start server
uv pip install vllm-mlx
python -m vllm_mlx.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8000
# Terminal 2: probe
uv run python docs/internal/research/20260330_macos_spike/verify_vllm_endpoints.py --port 8000 --server-name vllm-mlx
```

## Scripts → Questions Mapping

| Script | Questions | Requires |
|--------|-----------|----------|
| `verify_gpu_metrics.py` | Q1 (ioreg keys), Q2 (memory budget), Q3 (pyobjc wheels), Q4 (semantics) | macOS Apple Silicon |
| `verify_launchd.py` | Q7 (lifecycle), Q8 (ThrottleInterval), Q9 (KeepAlive) | macOS |
| `verify_logging.py` | Q10 (unified logging), Q16 (Metal fatal patterns) | macOS |
| `verify_vllm_endpoints.py` | Q5 (vllm-metal/vllm-mlx compatibility) | Running vLLM server |
| `verify_ram_portability.py` | Q11 (MemoryInfoProvider vs psutil) | Any OS |

## Questions NOT covered by scripts

| Question | Why | How to answer |
|----------|-----|---------------|
| Q6 (llama.cpp Metal perf) | Needs llama-bench + GGUF models | Run `llama-bench` manually with Q4_K_M/Q5_K_M/Q8_0 |
| Q12 (template dispatch) | Architecture design, not empirical | Code analysis on Day 3 |
| Q13-Q14 (ProcessController) | Architecture design | Code analysis on Day 3 |
| Q15 (DockerDriver scope) | Decision, not experiment | Discussion on Day 3 |
