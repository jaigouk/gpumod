# RLM and LangExtract Architecture Research

Research for spike `gpumod-c8l`: Knowledge base architecture for model configuration recommendations.

## Problem Statement

**Query**: "Can I run Qwen3-Coder-Next on 24GB VRAM?"

**Required reasoning chain**:

1. Model metadata → 80B total params, 3B activated (MoE)
2. MoE architecture → `--n-cpu-moe` available for expert offloading
3. Quant options → IQ2_XXS (26GB), Q2_K (29GB), etc.
4. KV cache → Flash attention prerequisite, cache-type quantization
5. System RAM → Can offload inactive experts to CPU
6. Live VRAM → Check actual available via `gpu_status`

**Problem with traditional RAG**: Retrieves fragments in isolation, misses dependency chains.

---

## RLM (Recursive Language Model)

**Source**: https://github.com/alexzhang13/rlm
**Paper**: Zhang et al. 2025

### Core Concept

Treat context as a **variable in a REPL environment** that the LLM can interact with programmatically, rather than stuffing everything into the prompt.

```python
# Traditional approach
llm.completion(prompt + context, model)

# RLM approach
rlm.completion(prompt, model)  # Context accessed via code exploration
```

### Implementation Pattern

````python
def run_rlm_agent(user_query, context_value, max_turns=10):
    # REPL globals - the "knowledge space"
    repl_globals = {
        "context": context_value,
        "llm_query": llm_query,  # Recursive sub-call
        "re": re,
    }

    messages = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {user_query}"}
    ]

    for turn in range(max_turns):
        response = llm.complete(messages)
        content = response.content

        # Check for final answer
        if "FINAL(" in content:
            return extract_answer(content)

        # Parse and execute code blocks
        code_match = re.search(r"```repl(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            output = execute_python_code(code, repl_globals)

            # Feed output back
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"REPL Output:\n{output}"})
````

### System Prompt Pattern

````
You are a Recursive Language Model (RLM).
You have access to a variable named `context` in your Python environment.

**TOOLS AVAILABLE:**
1. Python REPL: Write code in ```repl``` blocks
2. `llm_query(prompt)`: Recursive sub-call for focused analysis

**RULES:**
- DO NOT ask to see the full context. It is too big.
- DO write Python code to inspect, slice, search
- WHEN DONE: Output as `FINAL(your answer here)`
````

### Strengths

- **Dependency navigation**: LLM decides what to explore next based on findings
- **Persistent state**: Accumulates knowledge across turns
- **Recursive decomposition**: Can spawn sub-queries for focused analysis
- **Tool integration**: Natural fit for calling MCP tools from REPL

### For gpumod

```python
repl_globals = {
    "model_db": {...},           # Model metadata, MoE status, quants
    "flags_db": {                # Organized by driver (mirrors Driver pattern)
        "llamacpp": {...},       # --flash-attn, --n-cpu-moe, --cache-type-k
        "vllm": {...},           # --gpu-memory-utilization, --kv-cache-dtype
    },
    "quirks_db": {...},          # Model-specific settings (GLM-4.7, etc.)
    "calculate_vram": fn,        # VRAM estimation function
    "gpu_status": mcp_tool,      # Live VRAM from MCP
    "list_model_files": mcp_tool, # Returns driver_hint for format detection
    "llm_query": llm_query,      # Recursive sub-call
}
```

Example exploration:

````python
# LLM writes:
```repl
model = model_db["Qwen3-Coder-Next"]
print(f"Params: {model['total_params']}B, Active: {model['active_params']}B")
print(f"MoE: {model['is_moe']}")
````

# System returns:

# Params: 80B, Active: 3B

# MoE: True

# LLM continues:

```repl
if model['is_moe']:
    moe_flag = flags_db["llamacpp"]["--n-cpu-moe"]
    print(f"Can offload to system RAM: {moe_flag['description']}")
```

````

---

## LangExtract (Google)

**Source**: https://github.com/google/langextract

### Core Concept

Extract structured information from unstructured text with **source grounding** - every extraction traces back to its exact location in the source document.

### Strengths

- **Source grounding**: Maps every extraction to exact text span
- **Visual highlighting**: Can show where facts came from
- **Structured attributes**: Extracts entities with metadata
- **Example-driven**: Few-shot patterns for relationship extraction

### Limitations

- **Single-document focus**: Designed for within-document extraction
- **No cross-doc dependencies**: Cannot infer transitive relationships across sources
- **Static extraction**: Better for ingestion than runtime queries

### For gpumod

Best suited for **knowledge ingestion pipeline**:

```python
# Parse llama.cpp server README
extractions = langextract.extract(
    text=llamacpp_readme,
    schema={
        "flag": str,
        "type": str,
        "default": str,
        "description": str,
    },
    examples=[
        ("--flash-attn [on|off|auto]", {
            "flag": "--flash-attn",
            "type": "enum",
            "default": "auto",
            "description": "Enable Flash Attention",
        })
    ]
)

# Store with source attribution
for ext in extractions:
    flags_db[ext.flag] = {
        **ext.attributes,
        "source": "llama.cpp/tools/server/README.md",
        "source_span": ext.text_span,  # Exact location
    }
````

---

## Recommended Architecture: RLM as Optional Consulting Layer

**Key insight**: RLM uses multiple LLM calls with code execution — expensive. It should NOT be a gateway for all requests.

### Architecture Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Assistant (Claude Code)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
    │ Direct MCP  │    │ Direct MCP  │    │  RLM Consulting     │
    │ (routine)   │    │ (discovery) │    │  (complex queries)  │
    └─────────────┘    └─────────────┘    └─────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
    │ gpu_status  │    │ search_hf   │    │ "Can I run X on Y?" │
    │ switch_mode │    │ list_gguf   │    │ "Best settings for  │
    │ start/stop  │    │ gen_preset  │    │  model Z?"          │
    └─────────────┘    └─────────────┘    └─────────────────────┘
```

### When to Use Each Path

| Path               | Use Case                                                  | Cost        |
| ------------------ | --------------------------------------------------------- | ----------- |
| **Direct MCP**     | Routine operations: switch mode, check status, start/stop | 0 LLM calls |
| **Discovery MCP**  | Find models, list files, generate presets                 | 0 LLM calls |
| **RLM Consulting** | Complex queries requiring reasoning across knowledge      | N LLM calls |

### RLM Consulting Architecture (when invoked)

```
┌─────────────────────────────────────────────────────────────────┐
│              RLM Consulting Tool (mcp_tools.consult)            │
│                                                                 │
│  Invoked by: AI assistant when query requires reasoning         │
│  NOT a gateway — an optional tool alongside other MCP tools     │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RLM Query Engine                           │
│  - System prompt: explore via code, don't answer directly       │
│  - REPL globals: knowledge stores + MCP tool wrappers           │
│  - Max turns: configurable (default: 5)                         │
└─────────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │  Knowledge    │      │  MCP Tool     │      │  Helper       │
    │  Stores       │      │  Wrappers     │      │  Functions    │
    │               │      │               │      │               │
    │  model_db     │      │  gpu_status() │      │  calc_vram()  │
    │  flags_db     │      │  list_gguf()  │      │  check_fit()  │
    │  quirks_db    │      │  search_hf()  │      │               │
    └───────────────┘      └───────────────┘      └───────────────┘
```

### Query Routing: Direct vs. RLM

**Direct MCP (no LLM overhead)**:

```
User: "What's my current VRAM usage?"
→ AI calls gpu_status() directly
→ Returns: { total: 24000, used: 8500, free: 15500 }

User: "List GGUF files for unsloth/Qwen3-Coder-Next-GGUF"
→ AI calls list_gguf_files(repo_id) directly
→ Returns: { files: [...], count: 12 }

User: "Switch to code mode"
→ AI calls switch_mode("code") directly
→ Returns: { success: true }
```

**RLM Consulting (multi-LLM reasoning)**:

```
User: "Can I run Qwen3-Coder-Next on 24GB?"
→ AI recognizes complex query, calls consult(query)
→ RLM Turn 1: model_db["Qwen3-Coder-Next"] → 80B MoE
→ RLM Turn 2: flags_db["llamacpp"]["--n-cpu-moe"] → expert offloading
→ RLM Turn 3: list_gguf_files(...) → IQ2_XXS at 26GB
→ RLM Turn 4: gpu_status() → 22GB free
→ RLM Turn 5: FINAL(recommendation + sources)
→ Returns: { can_run: true, config: {...}, sources: [...] }
```

### Knowledge Population (optional LangExtract)

If structured knowledge stores are needed:

1. **Tier 1 (Core docs)**: Parse llama.cpp/vLLM official docs
   - Extract flags, types, defaults, descriptions
   - Store with source attribution

2. **Tier 2 (Curated)**: Parse Unsloth model pages, HF cards
   - Extract model metadata, quant recommendations
   - Store with version/date tracking

3. **Tier 3 (Community)**: Parse Reddit threads, GitHub discussions
   - Extract quirks, workarounds (e.g., GLM-4.7 repeat_penalty)
   - Flag as "community knowledge" with validation status

**Alternative (RLM-first)**: Skip structured extraction, let RLM search raw docs via `search_docs()` function in REPL.

---

## Knowledge Categories

### 1. Model Metadata

```python
{
    "repo_id": "unsloth/Qwen3-Coder-Next-GGUF",
    "total_params": 80_000_000_000,
    "active_params": 3_000_000_000,
    "is_moe": True,
    "context_length": 262144,
    "architecture": "qwen3next",
}
```

### 2. Quantization

```python
{
    "Q4_K_M": {
        "bits": 4,
        "size_multiplier": 0.5,
        "quality": "good",
        "vram_formula": "params * 0.5 + overhead",
    }
}
```

### 3. Memory Optimization (in flags_db["llamacpp"])

```python
{
    "--cache-type-k": {
        "options": ["f32", "f16", "bf16", "q8_0", "q4_0"],
        "default": "f16",
        "effect": "KV cache memory reduction",
        "prerequisite": "--flash-attn",
        "source": "llama.cpp/tools/server/README.md:L42",
    }
}
```

### 4. MoE Settings (in flags_db["llamacpp"])

```python
{
    "--n-cpu-moe": {
        "type": "int",
        "description": "Keep first N MoE layers on CPU",
        "use_case": "Offload inactive experts when GPU VRAM insufficient",
        "requires": "Sufficient system RAM",
        "source": "llama.cpp/tools/server/README.md:L87",
    }
}
```

### 5. Model Quirks

```python
{
    "GLM-4.7-Flash": {
        "issue": "Enters repetition loops with repeat_penalty enabled",
        "solution": "--repeat-penalty 1.0",
        "recommended_params": "--temp 0.7 --top-p 1.0 --min-p 0.01",
        "source": "huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/13",
        "tier": 3,
        "validated": True,
    }
}
```

### 6. flags_db Structure (organized by driver)

```python
flags_db = {
    "llamacpp": {
        "--flash-attn": {...},
        "--cache-type-k": {...},
        "--n-cpu-moe": {...},
    },
    "vllm": {
        "--gpu-memory-utilization": {...},
        "--kv-cache-dtype": {...},
        "--tensor-parallel-size": {...},
    }
}
```

Access pattern: `flags_db["llamacpp"]["--n-cpu-moe"]`

---

## Source Hierarchy

| Tier | Source Type             | Trust Level      | Update Frequency   |
| ---- | ----------------------- | ---------------- | ------------------ |
| 1    | llama.cpp docs          | Authoritative    | Track commits      |
| 1    | vLLM docs               | Authoritative    | Track releases     |
| 2    | Unsloth model pages     | Trusted          | Track page updates |
| 2    | HuggingFace model cards | Trusted          | Track repo updates |
| 3    | Reddit r/LocalLLaMA     | Needs validation | Manual curation    |
| 3    | GitHub discussions      | Needs validation | Manual curation    |

---

## Integration with Existing gpumod Architecture

### Alignment with docs/ARCHITECTURE.md

RLM fits as an **additional MCP tool**, not a replacement for existing tools:

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interfaces                             │
│           CLI · Interactive TUI · MCP Server                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      MCP TOOLS                                  │
├─────────────────────────────────────────────────────────────────┤
│  READ-ONLY:                      │  MUTATING:                   │
│  ├── gpu_status                  │  ├── switch_mode             │
│  ├── simulate_mode               │  ├── start_service           │
│  ├── search_hf_models            │  └── stop_service            │
│  ├── list_gguf_files             │                              │
│  ├── list_model_files            │  NEW (read-only):            │
│  ├── generate_preset             │  └── consult ← RLM engine    │
│  └── model_info                  │      (returns advice only)   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SERVICE LAYER (unchanged)                     │
│  ServiceManager → Drivers (VLLMDriver, LlamaCppDriver, ...)     │
└─────────────────────────────────────────────────────────────────┘
```

**Key boundary**: `consult` can only call read-only tools internally. It returns recommendations with suggested commands, but never executes mutating operations.

### RLM `consult` Tool Specification

```python
# New MCP tool: consult
async def consult(
    query: str,           # "Can I run Qwen3-Coder-Next on 24GB?"
    max_turns: int = 5,   # Limit LLM calls
    ctx: Context,
) -> dict:
    """
    Use RLM to answer complex queries requiring multi-step reasoning.

    SCOPE: Read-only consulting. Returns recommendations, not actions.

    When to use:
    - Query involves model + hardware + optimization tradeoffs
    - Query requires knowledge across multiple sources
    - Direct MCP tools don't provide enough context

    When NOT to use:
    - Simple lookups: use search_hf_models, list_gguf_files directly
    - Routine operations: use gpu_status, switch_mode directly
    """
```

### Tool Access Scope

RLM consult is **read-only** — it provides recommendations but does NOT execute actions.

| Tool               | Allowed | Reason                                           |
| ------------------ | ------- | ------------------------------------------------ |
| `gpu_status`       | ✅      | Read current VRAM state                          |
| `simulate_mode`    | ✅      | Simulate VRAM scenarios                          |
| `search_hf_models` | ✅      | Find models                                      |
| `list_gguf_files`  | ✅      | Get quant options, VRAM estimates                |
| `list_model_files` | ✅      | Detect model format, driver hint                 |
| `generate_preset`  | ✅      | Generate config (read-only, returns YAML string) |
| `switch_mode`      | ❌      | Mutating — user must execute                     |
| `start_service`    | ❌      | Mutating — user must execute                     |
| `stop_service`     | ❌      | Mutating — user must execute                     |

**Principle**: RLM consult returns a recommendation with suggested commands. The AI assistant (or user) decides whether to execute.

```python
# Example consult response
{
    "can_run": True,
    "recommendation": "Use IQ2_XXS with --n-cpu-moe 24",
    "suggested_commands": [
        "generate_preset(repo_id='unsloth/Qwen3-Coder-Next-GGUF', gguf_file='model-IQ2_XXS.gguf')",
        "switch_mode('qwen-code')",  # User/AI must decide to execute
    ],
    "sources": ["llama.cpp/tools/server/README.md:L87", "unsloth/Qwen3-Coder-Next-GGUF"]
}
```

### REPL Globals (when consult is invoked)

```python
repl_globals = {
    # Knowledge stores (populated from docs/community)
    "model_db": model_db,
    "flags_db": {                # Organized by driver
        "llamacpp": {...},       # --flash-attn, --n-cpu-moe
        "vllm": {...},           # --gpu-memory-utilization
    },
    "quirks_db": quirks_db,

    # READ-ONLY MCP tools (no mutating operations)
    "gpu_status": lambda: mcp_call("gpu_status"),
    "simulate_mode": lambda mode, **kw: mcp_call("simulate_mode", mode_id=mode, **kw),
    "search_hf_models": lambda **kw: mcp_call("search_hf_models", **kw),
    "list_gguf_files": lambda repo, **kw: mcp_call("list_gguf_files", repo_id=repo, **kw),
    "list_model_files": lambda repo, **kw: mcp_call("list_model_files", repo_id=repo, **kw),
    "generate_preset": lambda repo, gguf, **kw: mcp_call("generate_preset", repo_id=repo, gguf_file=gguf, **kw),

    # Helper functions
    "calculate_vram": calculate_vram,
    "llm_query": llm_query,  # Recursive sub-call for focused analysis (counts toward max_turns)

    # NOTE: switch_mode, start_service, stop_service are NOT exposed
    # RLM returns recommendations; execution is left to user/AI
}
```

**`llm_query` function**: Spawns a sub-RLM call with a focused prompt. Useful for analyzing a specific piece of knowledge without polluting the main conversation. Each sub-call counts toward `max_turns`.

Example RLM session:

````python
# User: "Can I run Qwen3-Coder-Next on my machine?"

# Turn 1: Check model
```repl
model = model_db.get("Qwen3-Coder-Next")
print(f"Model: {model}")
print(f"MoE: {model['is_moe']}, Active: {model['active_params']/1e9}B")
````

# Output: Model: {...}, MoE: True, Active: 3.0B

# Turn 2: Check available VRAM

```repl
status = gpu_status()
print(f"Available VRAM: {status['free_vram_mb']}MB")
print(f"Total VRAM: {status['total_vram_mb']}MB")
```

# Output: Available VRAM: 22000MB, Total VRAM: 24000MB

# Turn 3: Check quant options

```repl
files = list_gguf_files("unsloth/Qwen3-Coder-Next-GGUF")
for f in files['files'][:5]:
    print(f"{f['filename']}: {f['estimated_vram_mb']}MB")
```

# Output: model-IQ2_XXS.gguf: 26100MB, ...

# Turn 4: Check MoE offloading

```repl
if model['is_moe']:
    moe_flag = flags_db["llamacpp"]["--n-cpu-moe"]
    print(f"MoE offloading available: {moe_flag['description']}")
    print(f"Source: {moe_flag['source']}")
```

# Output: MoE offloading available: Keep first N MoE layers on CPU

# Turn 5: Final recommendation

FINAL({
"can_run": True,
"recommendation": "Use IQ2_XXS with --n-cpu-moe for expert offloading",
"estimated_vram": "~20GB with offloading",
"required_ram": "32GB+ system RAM recommended",
"sources": [
"unsloth/Qwen3-Coder-Next-GGUF",
"llama.cpp/tools/server/README.md:L87"
]
})

```

---

## Open Questions

### Security: REPL Code Execution

RLM executes LLM-generated Python code. Risks:
- **Arbitrary code execution**: LLM could generate malicious code
- **Resource exhaustion**: Infinite loops, memory bombs

**Sandboxing Solution: [sandbox-runtime (srt)](https://github.com/anthropic-experimental/sandbox-runtime)**

Anthropic's sandbox-runtime provides OS-level isolation for code execution:

| Platform | Backend | Description |
|----------|---------|-------------|
| Linux | bubblewrap | User namespace sandboxing via `bwrap` |
| macOS | sandbox-exec | Seatbelt profiles via `sandbox-exec` |

**Key restrictions:**
- **Network**: Allow-only (all network access blocked by default)
- **Filesystem read**: Deny-only list for sensitive paths
- **Filesystem write**: Allow-only (writes blocked except explicit paths)
- **Process**: Isolated namespace, no access to host processes

**Usage pattern:**
```bash
# Run Python script in sandbox
srt python script.py

# With explicit permissions
srt --network=allow --write=/tmp python script.py
```

**Implementation for RLM:**
```python
import subprocess

def execute_sandboxed(code: str, timeout: float = 5.0) -> str:
    """Execute code in sandbox-runtime."""
    result = subprocess.run(
        ["srt", "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr
```

**Additional mitigations (defense in depth):**
1. Restricted globals (only expose safe functions)
2. Code execution timeout (default: 5 seconds per block)
3. No `import`, `exec`, `eval`, `open` in REPL globals
4. AST validation before execution (optional, srt handles most risks)

### Error Handling

| Scenario | Behavior |
|----------|----------|
| max_turns exhausted | Return partial findings + "incomplete" flag |
| Code execution error | Feed error back to LLM, let it retry |
| MCP tool failure | Feed error back, LLM can try alternative |
| Missing knowledge | LLM can call MCP tools to fetch live data |

### LLM Model Selection

Options for RLM engine:
- **Haiku**: Fast, cheap, good for simple queries (default)
- **Sonnet**: Better reasoning, higher cost
- **Configurable**: Let user set via environment variable

### Storage & Persistence

Knowledge stores options:
- **JSON files**: Simple, version-controllable, slow for large stores
- **SQLite**: Fast queries, single file, already used by gpumod
- **Hybrid**: JSON for static knowledge, SQLite for runtime cache

**Recommendation**: Start with JSON files in `~/.config/gpumod/knowledge/`

### Caching

- **consult results**: No caching (queries are contextual)
- **Knowledge stores**: Reload on startup, manual refresh command
- **MCP tool results**: Reuse existing MCP caching (e.g., HuggingFaceSearcher)

---

## Critical Insight: Knowledge Is Not Static

The earlier analysis assumed pre-populated knowledge stores (model_db, flags_db). This is a flawed assumption.

### The Real Problem

When a user asks "Can I run Qwen3-235B-Instruct on 24GB?", we need:

```
1. Search HuggingFace → find repo_id
2. Fetch model card (raw markdown) → parse architecture info
3. Fetch config.json → extract total_params, is_moe, context_length
4. Fetch llama.cpp README → find relevant flags for this architecture
5. Cross-reference with Unsloth recommendations
6. Combine with live VRAM check
7. Generate recommendation
```

**Key facts:**
- `model_db` is NOT pre-populated — new models appear constantly
- `flags_db` is NOT static — llama.cpp/vLLM flags are updated with each release
- Claude's training data does NOT have current flag knowledge
- Knowledge must come from **validated sources**, not random blog posts

### Why "Just Use Claude" Fails

My earlier experiment (rlm_experiment.py) concluded "skip RLM, just use Claude." This was wrong because:

1. **Claude doesn't have current knowledge**: llama.cpp adds/changes flags frequently. Claude's training cutoff means it lacks recent flags like `--n-cpu-moe`, `--cache-type-k` options, etc.

2. **Source quality matters**: If Claude web searches, it might find unreliable blog posts. Validated sources include:
   - Official READMEs (llama.cpp/tools/server/README.md)
   - Unsloth docs (https://unsloth.ai/docs/models/*)
   - HuggingFace model cards from trusted uploaders

3. **Extraction still required**: Raw markdown docs need parsing to extract structured information (flag types, defaults, dependencies).

### Revised Architecture Options

Given this complexity, three viable approaches:

#### Option A: Runtime Fetching + Claude Reasoning

```
Query → Fetch raw docs → Claude reasons over fetched content → Answer
```

**Flow:**
```python
# MCP tool: fetch_model_info(repo_id)
1. Fetch config.json from HuggingFace
2. Fetch README.md from repo
3. Return structured JSON + raw markdown

# MCP tool: fetch_driver_docs(driver="llamacpp")
1. Fetch llama.cpp server README
2. Return raw markdown

# Claude receives raw docs in context, reasons directly
```

**Pros:**
- Simple implementation (just fetch, no parsing)
- Claude can reason over raw markdown
- No separate RLM calls

**Cons:**
- Large context window usage (full READMEs)
- Claude may miss structured details in dense docs
- No persistent knowledge cache

#### Option B: Ingestion Pipeline + Cached Knowledge

```
[Periodic] Ingest docs → Extract structured data → Store in flags_db
[Runtime]  Query → Lookup cached knowledge → Answer
```

**Flow:**
```python
# Ingestion (daily/weekly cron or manual)
1. Fetch llama.cpp README → LangExtract → flags_db["llamacpp"]
2. Fetch vLLM docs → LangExtract → flags_db["vllm"]
3. Fetch popular model cards → extract metadata → model_db

# Runtime
1. User query → lookup flags_db/model_db → answer
2. Unknown model? Trigger on-demand fetch + extraction
```

**Pros:**
- Fast lookups (pre-extracted)
- Consistent structure (LangExtract schema)
- Source attribution preserved

**Cons:**
- Maintenance burden (ingestion pipeline)
- Stale data between ingestion runs
- Complex for new/unknown models

#### Option C: Hybrid RLM with Live Fetching

```
Query → RLM with fetch tools → Explore raw docs programmatically → Answer
```

**Flow:**
```python
repl_globals = {
    # Live fetch tools (NOT pre-populated stores)
    "fetch_model_card": lambda repo: fetch_hf_readme(repo),
    "fetch_config": lambda repo: fetch_hf_config_json(repo),
    "fetch_driver_docs": lambda driver: fetch_official_readme(driver),
    "web_search": lambda query, domains: search_validated_sources(query, domains),

    # Live MCP tools
    "gpu_status": gpu_status,
    "list_gguf_files": list_gguf_files,

    # Helpers
    "parse_json": json.loads,
    "search_text": lambda text, pattern: re.findall(pattern, text),
}
```

**Example RLM session:**
```python
# Turn 1: Find the model
```repl
config = fetch_config("Qwen/Qwen3-235B-Instruct-GGUF")
print(f"Params: {config.get('num_parameters', 'unknown')}")
print(f"Architecture: {config.get('architectures', ['unknown'])[0]}")
```

# Turn 2: Check if MoE
```repl
if config.get('num_experts'):
    print(f"MoE model: {config['num_experts']} experts")
    # Need to fetch MoE-specific flags
    docs = fetch_driver_docs("llamacpp")
    moe_section = search_text(docs, r'--n-cpu-moe.*?(?=\n--|\Z)')
    print(f"MoE flag: {moe_section}")
```

# Turn 3: Get validated recommendations
```repl
# Search Unsloth docs for this model
unsloth = web_search("Qwen3-235B site:unsloth.ai/docs")
print(f"Unsloth guide: {unsloth}")
```
```

**Pros:**
- Always current (fetches live docs)
- Programmatic exploration of raw text
- Can target validated sources

**Cons:**
- Multiple LLM calls (expensive)
- Sandbox paradox (srt blocks network for fetched code)
- Latency from live fetches

### The Sandboxing Paradox (Revisited)

Option C has a fundamental problem: if RLM code runs in srt sandbox, it CANNOT make network requests.

**Solutions:**
1. **Pre-fetch before sandbox**: Fetch all potentially needed docs, pass as context to sandboxed RLM
2. **No sandbox**: Trust the code generation (risky)
3. **Tool-based fetch**: RLM doesn't fetch directly; it calls "request_fetch(url)" which queues fetches for the outer loop

**Tool-based fetch pattern:**
```python
# RLM code writes:
```repl
request_fetch("llamacpp_readme", "https://raw.githubusercontent.com/.../README.md")
PAUSE()  # Signal to outer loop: execute fetches, then continue
```

# Outer loop (not sandboxed):
1. See PAUSE() with pending fetch requests
2. Execute fetches
3. Inject results into repl_globals
4. Resume RLM with new data available
```

### Recommendation

**For spike gpumod-c8l, use Option C** (Hybrid RLM with Live Fetching):

The `rlm_experiment.py` tests verified:
1. **Function injection works**: Both `setup_code` (string) and `repl.globals` (real functions) injection work
2. **Function whitelisting works**: Undefined functions raise `NameError`, so we can expose ONLY read-only MCP tools
3. **Multi-step reasoning works**: RLM can chain calls across multiple functions

**Architecture:**
```python
repl_globals = {
    # READ-ONLY MCP tools (whitelisted)
    "gpu_status": gpu_status,              # Live VRAM check
    "list_gguf_files": list_gguf_files,    # Quant options + VRAM estimates
    "fetch_model_config": fetch_config,     # config.json from HuggingFace
    "fetch_driver_docs": fetch_driver_docs, # Official llama.cpp/vLLM README
    "search_hf_models": search_hf_models,   # Find models

    # NOT EXPOSED (mutating operations):
    # switch_mode, start_service, stop_service
    # These raise NameError if LLM tries to call them
}
```

**Why Option C over Option A:**
- Option A (Claude + fetched docs) uses large context windows for raw READMEs
- Option C uses programmatic exploration — only fetch what's needed
- RLM can search, slice, and cross-reference docs iteratively
- Function whitelisting ensures RLM can only read, never mutate

**Sandboxing approach**: Use tool-based fetch pattern. RLM code requests fetches via `request_fetch()`, outer loop executes them, injects results back. This allows srt sandboxing for the code execution while permitting network access from the orchestrator.

**Key principle**: Fetch from validated sources, expose only read-only tools.

---

## Next Steps

1. [ ] Add `fetch_model_config(repo_id)` MCP tool (config.json from HuggingFace)
2. [ ] Add `fetch_model_readme(repo_id)` MCP tool (README.md from HuggingFace)
3. [ ] Add `fetch_driver_docs(driver)` MCP tool (official llama.cpp/vLLM README)
4. [ ] Implement RLM `consult` MCP tool with function whitelisting
5. [ ] Add tool-based fetch pattern (`request_fetch()` + `PAUSE()`) for sandbox compatibility
6. [ ] Test with real query: "Can I run Qwen3-235B-Instruct on 24GB?"
7. [ ] Evaluate reasoning quality and iteration count
8. [ ] Consider ingestion pipeline (Option B) for frequently-accessed knowledge as optimization

---

## References

- [RLM Paper](https://github.com/alexzhang13/rlm) - Zhang et al. 2025
- [LangExtract](https://github.com/google/langextract) - Google
- [sandbox-runtime (srt)](https://github.com/anthropic-experimental/sandbox-runtime) - Anthropic sandboxing
- [llama.cpp Server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [vLLM Optimization](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [HuggingFace MoE Offload Guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
```
