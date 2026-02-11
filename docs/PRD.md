---
title: "PRD: Model Knowledge Base & RLM Consult Tool"
epic: gpumod-rch
feature: gpumod-pyf
status: draft
date: 2026-02-11
---

# PRD: Model Knowledge Base & RLM Consult Tool

## 1. Problem Statement

Community knowledge about ML model configurations is scattered across Reddit
posts, HuggingFace model cards, Unsloth docs, and Discord forums. When a user
asks "Can I run Qwen3-235B-Instruct on 24GB?", answering requires chaining
multiple data sources:

1. Model metadata (params, MoE status, architecture)
2. Quantization options (GGUF variants, VRAM estimates)
3. Driver-specific flags (llama.cpp `--n-cpu-moe`, vLLM `--gpu-memory-utilization`)
4. Live GPU state (current VRAM availability)
5. Community quirks (e.g., GLM-4.7 needs `repeat_penalty=1.0`)

No single MCP tool can answer this. Direct MCP tools return isolated data.
The user or AI assistant must manually chain calls and reason across results.

## 2. Solution: RLM Consulting Layer

Add an **optional** `consult` MCP tool that uses
[Recursive Language Models (RLM)](https://github.com/alexzhang13/rlm) to answer
complex queries requiring multi-step reasoning. RLM treats context as a
**variable in a REPL environment** that the LLM explores programmatically.

### Architecture Principle: Three Tool Tiers

```
Tier 1: Direct MCP (0 LLM calls)     → gpu_status, switch_mode, start/stop
Tier 2: Discovery MCP (0 LLM calls)  → search_hf, list_gguf, fetch_config
Tier 3: RLM Consulting (N LLM calls) → consult (complex reasoning)
```

The `consult` tool is **read-only**. It returns recommendations with source
citations but never executes mutating operations.

## 3. Technical Design

### 3.1 Custom RLM Environment

Use RLM's `NonIsolatedEnv` base class (not manual globals injection) for a
clean, testable custom environment:

```python
from rlm.environments.base_env import NonIsolatedEnv
from rlm.core.types import REPLResult

class GpumodConsultEnv(NonIsolatedEnv):
    """Custom RLM environment with whitelisted gpumod MCP tools.

    Only read-only tools are exposed. Mutating operations
    (switch_mode, start_service, stop_service) raise NameError.
    """

    def __init__(self, tool_wrappers: dict, **kwargs):
        self._tool_wrappers = tool_wrappers
        super().__init__(**kwargs)

    def setup(self):
        self._namespace = {
            # Read-only MCP tools (whitelisted)
            'gpu_status': self._tool_wrappers['gpu_status'],
            'list_gguf_files': self._tool_wrappers['list_gguf_files'],
            'fetch_model_config': self._tool_wrappers['fetch_model_config'],
            'fetch_driver_docs': self._tool_wrappers['fetch_driver_docs'],
            'search_hf_models': self._tool_wrappers['search_hf_models'],
            'simulate_mode': self._tool_wrappers['simulate_mode'],
            'generate_preset': self._tool_wrappers['generate_preset'],
            # Standard library helpers
            'json': __import__('json'),
            're': __import__('re'),
        }

    def load_context(self, payload):
        self._namespace['context'] = payload

    def execute_code(self, code: str) -> REPLResult:
        # Execute in restricted namespace with timeout
        ...
```

### 3.2 Consult MCP Tool

```python
async def consult(
    query: str,
    ctx: Context,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Multi-step reasoning for complex GPU/model questions.

    Uses RLM to programmatically explore model metadata, driver docs,
    and VRAM constraints. Returns recommendations, never executes actions.
    """
```

### 3.3 Response Format

```json
{
  "can_run": true,
  "recommendation": "Use IQ2_XXS with --n-cpu-moe 24 for expert offloading",
  "reasoning_steps": [
    "Fetched model config: 80B MoE with 8 experts",
    "Checked GGUF files: IQ2_XXS at 26.1GB",
    "Checked GPU: 22GB free VRAM",
    "Found --n-cpu-moe flag for expert offloading"
  ],
  "suggested_commands": [
    "generate_preset(repo_id='unsloth/Qwen3-Coder-Next-GGUF', gguf_file='model-IQ2_XXS.gguf')"
  ],
  "sources": [
    "huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/config.json",
    "llama.cpp/tools/server/README.md"
  ],
  "turns_used": 4
}
```

### 3.4 REPL Globals (Whitelisted Read-Only Tools)

| Tool                | Type      | Purpose                              |
| ------------------- | --------- | ------------------------------------ |
| `gpu_status`        | Read-only | Live VRAM state                      |
| `list_gguf_files`   | Read-only | GGUF quants + VRAM estimates         |
| `fetch_model_config`| Read-only | config.json (architecture, MoE, ctx) |
| `fetch_driver_docs` | Read-only | Official llama.cpp/vLLM README       |
| `search_hf_models`  | Read-only | Find models on HuggingFace           |
| `simulate_mode`     | Read-only | Simulate VRAM scenarios              |
| `generate_preset`   | Read-only | Generate YAML config                 |

**Blocked (raise NameError):** `switch_mode`, `start_service`, `stop_service`

### 3.5 Security Model

1. **Function whitelisting**: Only explicitly exposed functions are callable
2. **Timeout**: 5s per code block, 60s total per consult call
3. **Max turns**: Configurable (default 5, max 10)
4. **No imports**: `import`, `exec`, `eval`, `open` blocked via AST check
5. **Read-only**: Consult returns advice; user/AI decides to execute

### 3.6 Dependencies

#### fetch_driver_docs (prerequisite)

New MCP tool to fetch official documentation:

```python
async def fetch_driver_docs(
    driver: str,            # "llamacpp" or "vllm"
    ctx: Context,
    version: str | None,    # None = auto-detect from system
    section: str | None,    # Filter to specific section
) -> dict[str, Any]:
```

- Auto-detects installed version (`llama-server --version`, `pip show vllm`)
- Fetches version-specific README from GitHub/docs
- 24h TTL cache
- Fallback URL chain for moved docs

## 4. Knowledge Sources

See `docs/research/knowledge_sources.md` for the full catalog.

| Tier | Source                     | Fetch Strategy | TTL    |
| ---- | -------------------------- | -------------- | ------ |
| 1    | llama.cpp/vLLM docs        | Live fetch     | 24h    |
| 2    | Unsloth, MoE guides        | Cached         | 7 days |
| 3    | HF config.json, model APIs | API fetch      | 1h     |
| 4    | Community quirks, VRAM rules| Manual curated | Manual |

## 5. Implementation Plan

### Phase 1: Foundation (gpumod-a6n)
- Implement `DriverDocsFetcher` class
- Add `fetch_driver_docs` MCP tool
- Version detection for llamacpp and vllm
- Tests with mocked HTTP responses

### Phase 2: RLM Core (gpumod-pyf)
- Create `src/gpumod/rlm/` module
- Implement `GpumodConsultEnv` (NonIsolatedEnv subclass)
- Implement `RLMOrchestrator` (manages RLM lifecycle)
- Add `consult` MCP tool to mcp_tools.py
- Register tool with MCP server
- Tests with mocked RLM responses

### Phase 3: Integration Testing (gpumod-kov)
- Test with real queries (requires API key)
- Validate reasoning quality and turn count
- Document results

### Phase 4: Documentation (gpumod-111)
- Update ARCHITECTURE.md with discovery + consult layers
- Add ADR for RLM architecture choice
- Document security model

### Phase 5: Maintenance (gpumod-wdd, gpumod-b9x)
- Periodic knowledge source validation
- Optional sandbox-compatible fetch pattern

## 6. Success Criteria

| Metric                | Target                        |
| --------------------- | ----------------------------- |
| consult tool works    | Registered in MCP, callable   |
| Function whitelisting | Mutating ops raise NameError  |
| Reasoning quality     | Correct conclusion on 4 test queries |
| Turn efficiency       | < 5 turns average             |
| Response time         | < 30s per query               |
| Security              | No arbitrary code execution   |

## 7. Out of Scope (v1)

- Automated knowledge ingestion pipeline (LangExtract)
- Full srt sandboxing (defer to gpumod-b9x)
- Web scraping for community knowledge
- Multi-GPU support
- Training/fine-tuning recommendations

## 8. References

- [RLM Paper](https://arxiv.org/abs/2512.24601) - Zhang et al. 2025
- [RLM Library](https://github.com/alexzhang13/rlm) - alexzhang13/rlm
- [RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/) - Alex Zhang
- [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime) - Anthropic
- [Research: RLM Architecture](docs/research/rlm_langextract.md) - Spike findings
- [Research: Knowledge Sources](docs/research/knowledge_sources.md) - Validated sources
- [Research: RLM Experiment](docs/research/rlm_experiment.py) - Function injection tests
