# Research: KV Cache Estimation Redesign -- Linear to Compound Formula

**Date:** 2026-05-24
**Spike Ticket:** gpumod-cf8
**Status:** Final
**Author:** researcher (gpumod-cf8)
**Prior art:** `docs/research/20260419_triattention_viability.md` section A

## Summary

gpumod's current KV cache estimation is strictly linear:
`total_vram = base_vram + (context_size / 1000) * kv_cache_per_1k_tokens_mb`. This
formula treats all transformer layers identically and assumes KV cache scales
linearly with context. This is wrong for hybrid-attention models (Gemma 3,
Gemma 3n, Gemma 4) where sliding-window layers cap their KV cache at the window
size, and for models with KV sharing where multiple layers reuse the same K/V
tensors.

The compound formula proposed here overestimates KV cache by **73-88% for
Gemma models** while producing identical results for dense models (Qwen3, Llama
3.3), making it a strict generalization that can replace the current formula
without breaking backward compatibility.

**Recommendation:** Adopt option (a) -- keep the existing scalar
`kv_cache_per_1k_tokens_mb` for backward compatibility and add a new structured
`kv_cache_profile` field to `ModelInfo`. Implement a two-phase rollout: first
add the profile-aware estimation path, then backfill existing models.

## Table of Contents

1. [Step 1: Call Site Audit](#step-1-call-site-audit)
2. [Step 2: Compound Formula Derivation](#step-2-compound-formula-derivation)
3. [Step 3: Persistence Shape](#step-3-persistence-shape)
4. [Step 4: Fetcher Refactor Prototype](#step-4-fetcher-refactor-prototype)
5. [Step 5: Reference Values](#step-5-reference-values)
6. [Step 6: Preflight Update Sketch](#step-6-preflight-update-sketch)
7. [Step 7: Deliverables](#step-7-deliverables)

---

## Step 1: Call Site Audit

All consumers of `kv_cache_per_1k_tokens_mb` identified by grepping `src/`
(verified 2026-05-24):

### Producer (1 site)

| File | Line | Role | What it needs |
|------|------|------|---------------|
| `fetchers/huggingface.py` | 263-299 | **Compute** | `_estimate_kv_cache_per_1k()` -- the linear formula. Inputs: `num_layers`, `hidden_size`, `num_kv_heads`, `num_attention_heads`. Returns `int` (MB). |

### Schema / definition (2 sites)

| File | Line | Role |
|------|------|------|
| `models.py` | 179 | `ModelInfo.kv_cache_per_1k_tokens_mb: int \| None` -- field definition |
| `models.py` | 207 | `PresetConfig.kv_cache_per_1k: int \| None` -- preset config field |

### Storage (4 sites)

| File | Line | Role |
|------|------|------|
| `db.py` | 91 | SQL schema: `kv_cache_per_1k_tokens_mb INTEGER` column in `models` table |
| `db.py` | 222 | `_row_to_model_info()` -- reads column into `ModelInfo` |
| `db.py` | 599 | `insert_model()` -- INSERT statement column list |
| `db.py` | 609 | `insert_model()` -- parameter binding |

### Consumer: VRAM estimation (3 sites)

| File | Line | Role | What it needs |
|------|------|------|---------------|
| `registry.py` | 129 | Docstring documents the formula | N/A |
| `registry.py` | 161 | Guard: `if model.kv_cache_per_1k_tokens_mb is not None` | Boolean check |
| `registry.py` | 162 | **Core consumer**: `kv_addition = int((context_size / 1000) * model.kv_cache_per_1k_tokens_mb)` | Needs either scalar or compound function |

### Consumer: Display (3 sites)

| File | Line | Role | What it needs |
|------|------|------|---------------|
| `cli_model.py` | 34 | JSON serialization: `"kv_cache_per_1k_tokens_mb": model.kv_cache_per_1k_tokens_mb` | Any serializable value |
| `cli_model.py` | 133-134 | Table column: `str(row["kv_cache_per_1k_tokens_mb"])` | Displayable string |
| `cli_model.py` | 184 | Panel output: `f"[bold]KV/1K:[/bold] {model.kv_cache_per_1k_tokens_mb or '-'} MB"` | Displayable value |
| `mcp_resources.py` | 175 | MCP resource: `f"- **KV Cache per 1k tokens:** {model.kv_cache_per_1k_tokens_mb or '-'} MB"` | Displayable value |

### Consumer: Preset loading (from presets/*.yaml)

21 preset files contain `kv_cache_per_1k: <int>` values. These are loaded via
`PresetConfig.kv_cache_per_1k` and used to populate `Service` / `ModelInfo`
during preset registration. Preset values are static overrides, not computed.

### Consumer: Test files (15+ sites)

Test files in `tests/unit/` and `tests/integration/` reference the field in
fixtures, assertions, and mock data. These will need updating if the field
type changes, but NOT if we add a new field alongside.

### Consumer summary

The critical consumer is `registry.py:162` -- the VRAM estimation formula.
All other consumers are either storage (passthrough), display (format for
humans), or tests. This means the migration can be done with minimal blast
radius: change the estimation logic in `registry.py`, extend the fetcher,
and add the new field. Display consumers can optionally show the profile
alongside the legacy scalar.

---

## Step 2: Compound Formula Derivation

### Current formula (linear, layer-homogeneous)

```
kv_per_1k = 2 * num_layers * num_kv_heads * head_dim * 2 * 1000 / 1024^2
total_kv  = (ctx / 1000) * kv_per_1k
```

- `2`: K + V tensors
- `num_layers`: all layers counted equally
- `num_kv_heads`: same for all layers
- `head_dim`: same for all layers
- `2`: bytes per fp16 element
- `1000`: tokens per 1K batch
- `1024^2`: bytes to MB

### Compound formula (layer-type-aware, non-linear)

```
per_tok_sliding = kv_factor * num_kv_heads * head_dim * bytes_per_elem
per_tok_global  = kv_factor * global_kv_heads * global_head_dim * bytes_per_elem

unique_sliding = n_sliding_layers - shared_sliding_layers
unique_global  = n_global_layers  - shared_global_layers

KV_total(ctx) = unique_sliding * min(ctx, sliding_window) * per_tok_sliding
              + unique_global  * min(ctx, triattn_budget)  * per_tok_global
```

Where:
- `kv_factor` = 1 if `attention_k_eq_v` else 2 (K=V halves storage)
- `global_kv_heads` = `num_global_key_value_heads` if present, else `num_kv_heads`
- `global_head_dim` = `global_head_dim` if present, else `head_dim`
- `shared_*_layers` = layers that reuse KV from an upstream layer of the same type
  (last `num_kv_shared_layers` layers, distributed proportionally across types)
- `triattn_budget` = TriAttention fixed key budget (future, currently infinity)

### Per-architecture reduction proofs

**Case 1: Dense (current behavior)**
- `layer_types` = all `full_attention`
- `sliding_window` = None (treated as infinity)
- `num_kv_shared_layers` = 0
- `triattn_budget` = None (infinity)

```
unique_global = num_layers, unique_sliding = 0
KV(ctx) = num_layers * min(ctx, inf) * 2 * num_kv_heads * head_dim * 2
        = num_layers * ctx * 2 * num_kv_heads * head_dim * 2
        = current formula * ctx / 1000 (when expressed in MB/1K)
```
MATCHES current formula. Verified numerically for Qwen3-32B and Llama-3.3-70B.

**Case 2: Sliding-window only (all layers use sliding attention)**
- `layer_types` = all `sliding_attention`
- `sliding_window` = W tokens

```
unique_sliding = num_layers
KV(ctx) = num_layers * min(ctx, W) * per_tok_sliding
```
At `ctx <= W`, this equals the linear formula. At `ctx > W`, KV is capped at
`num_layers * W * per_tok` -- constant regardless of context growth.

**Case 3: Hybrid (sliding + global)**
- `layer_types` = mix of `sliding_attention` and `full_attention`
- `sliding_window` = W

```
KV(ctx) = unique_sliding * min(ctx, W) * per_tok_sliding
        + unique_global  * ctx * per_tok_global
```
KV grows linearly with context only from the global layers. Sliding layers
contribute a constant ceiling once `ctx > W`. When `W = inf`, all layers are
effectively global and the formula reduces to Case 1.

**Case 4: Hybrid + KV sharing (Gemma 3n style)**
- `num_kv_shared_layers` = S > 0

Same as Case 3 but `unique_sliding` and `unique_global` are reduced by
the shared layer counts. When S = 0, reduces to Case 3.

**Case 5: Hybrid + asymmetric head_dim (Gemma 4 26B/31B style)**
- `global_head_dim` != `head_dim`
- `num_global_key_value_heads` != `num_kv_heads`

Global layers use `per_tok_global` with different dimensions. Sliding layers
use `per_tok_sliding` with base dimensions. When `global_head_dim = head_dim`
and `global_kv_heads = kv_heads`, reduces to Case 3/4.

**Case 6: Hybrid + TriAttention budget (future)**
- `triattn_budget` = B

```
KV(ctx) = unique_sliding * min(ctx, W) * per_tok_sliding
        + unique_global  * min(ctx, B) * per_tok_global
```
Once `ctx > B`, global-layer KV is also capped. When B = inf, reduces to
Case 3/4/5.

### Backward-compatibility verification

Numerically verified for all 4 reference models: the compound formula with
`sliding_window=None`, `triattn_budget=None`, `num_kv_shared_layers=0` produces
identical results to the current `_estimate_kv_cache_per_1k()` at ctx = 1K, 8K,
32K, 128K. See PoC output in Step 5.

---

## Step 3: Persistence Shape

### Option (a): Additive -- keep scalar, add structured profile

```python
class KVCacheProfile(BaseModel):
    """Structured KV cache estimation data for hybrid-attention models."""

    model_config = ConfigDict(extra="forbid")

    num_sliding_layers: int = 0
    num_global_layers: int = 0
    num_kv_shared_layers: int = 0
    sliding_window: int | None = None       # None = no sliding window
    head_dim: int = 128
    global_head_dim: int | None = None       # None = same as head_dim
    num_kv_heads: int = 1
    num_global_kv_heads: int | None = None   # None = same as num_kv_heads
    attention_k_eq_v: bool = False
    triattn_budget: int | None = None        # None = no TriAttention
    # Redundant but useful for display and backward compat
    kv_per_1k_at_inf: int | None = None      # linear-equivalent rate

class ModelInfo(BaseModel):
    # ... existing fields unchanged ...
    kv_cache_per_1k_tokens_mb: int | None = None   # KEPT for backward compat
    kv_cache_profile: KVCacheProfile | None = None  # NEW, optional
```

**DB migration:**
- Add `kv_cache_profile TEXT` column to `models` table (JSON-serialized)
- One `ALTER TABLE ... ADD COLUMN` migration (same pattern as v2->v3)
- Existing rows have `kv_cache_profile = NULL` -- estimation falls back to
  `kv_cache_per_1k_tokens_mb` scalar

**Estimation logic change in `registry.py`:**
```python
# Pseudocode for new estimate_vram()
if model.kv_cache_profile is not None:
    kv_addition = compute_compound_kv(model.kv_cache_profile, context_size)
elif model.kv_cache_per_1k_tokens_mb is not None:
    kv_addition = int((context_size / 1000) * model.kv_cache_per_1k_tokens_mb)
else:
    kv_addition = 0
total = base_vram + kv_addition
```

### Option (b): Replace scalar with struct

Replace `kv_cache_per_1k_tokens_mb` entirely with `kv_cache_profile`.

**Pros:**
- Cleaner schema -- one source of truth
- No ambiguity about which field to use

**Cons:**
- Breaking change to `ModelInfo` -- all 15+ test sites need updating
- Breaking change to `PresetConfig` -- all 21 preset YAML files need updating
- CLI and MCP display code needs rewriting
- DB column rename or migration that drops and re-creates data
- External tools that parse `ModelInfo` JSON break

### Recommendation: Option (a)

**Rationale:**
1. **Zero migration cost for existing data.** The 21 preset files and all test
   fixtures continue to work unchanged. New models populated via the fetcher
   get both fields; old models use the scalar fallback.
2. **Gradual rollout.** The fetcher can start populating `kv_cache_profile`
   immediately; `registry.py` can prefer it when available. No flag day.
3. **Display backward compat.** CLI and MCP already show `kv_cache_per_1k_tokens_mb`.
   The profile is additional detail, not a replacement for the summary metric.
4. **DB migration is trivial.** One `ALTER TABLE models ADD COLUMN kv_cache_profile TEXT`
   with NULL default. Same pattern used for `preflight_required` and `compat`
   columns (see `db.py:137-152`).

**Estimated migration cost:**
- `models.py`: +15 lines (new Pydantic model + field)
- `db.py`: +10 lines (column, read, write)
- `registry.py`: +20 lines (compound estimation logic)
- `fetchers/huggingface.py`: +40 lines (parse config.json, build profile)
- Tests: +30 lines (new test cases for compound formula)
- Presets: 0 changes
- CLI/MCP: 0 required changes (optional: show profile detail)

**Total: ~115 lines added, 0 lines changed in existing code.**

---

## Step 4: Fetcher Refactor Prototype

The current `HuggingFaceFetcher.fetch()` uses `huggingface_hub.model_info()` which
returns a partial config view via `info.config`. This dict contains the top-level
config fields (`num_hidden_layers`, `hidden_size`, `num_attention_heads`,
`num_key_value_heads`) but does NOT reliably surface:

- `layer_types` (only present in Gemma 3n, not Gemma 3)
- `sliding_window_pattern` (not in config.json; derived by transformers)
- `sliding_window` (present but not in `info.config` for all models)
- `num_kv_shared_layers` (Gemma 3n/4 specific)
- `global_head_dim` (Gemma 4 specific)
- `num_global_key_value_heads` (Gemma 4 specific)
- `attention_k_eq_v` (Gemma 4 specific)
- `text_config` nesting (multimodal models like Gemma 3/3n/4 nest text config)

### Solution: additional `hf_hub_download` call for `config.json`

`hf_hub_download` is already a dependency (from `huggingface_hub`). It downloads
and caches `config.json` locally. This is a single HTTP GET with local caching,
adding ~50ms latency on cache miss and 0ms on cache hit.

### Prototype code

```python
# In fetchers/huggingface.py -- proposed addition

import json
from huggingface_hub import hf_hub_download

def _fetch_raw_config(self, model_id: str) -> dict | None:
    """Fetch and parse raw config.json from HuggingFace Hub.

    Uses hf_hub_download which caches locally. Falls back to None
    on any error (gated repos, network issues, missing file).
    """
    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _build_kv_cache_profile(self, config: dict) -> KVCacheProfile | None:
    """Build KVCacheProfile from raw config.json.

    Handles both top-level configs (Qwen, Llama) and nested
    text_config (Gemma 3, 3n, 4).
    """
    # Resolve text_config nesting for multimodal models
    tc = config.get("text_config", config)

    num_layers = tc.get("num_hidden_layers")
    num_heads = tc.get("num_attention_heads")
    num_kv_heads = tc.get("num_key_value_heads", num_heads)
    hidden_size = tc.get("hidden_size")
    head_dim = tc.get("head_dim")

    if num_layers is None or num_heads is None or hidden_size is None:
        return None

    if head_dim is None and num_heads > 0:
        head_dim = hidden_size // num_heads

    # Determine layer types
    layer_types = tc.get("layer_types")
    sliding_window = tc.get("sliding_window")
    sliding_window_pattern = tc.get("sliding_window_pattern")

    # If no explicit layer_types but has sliding_window_pattern, derive
    if layer_types is None and sliding_window_pattern is not None:
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern)
            else "full_attention"
            for i in range(num_layers)
        ]

    # If no layer_types and no pattern but has sliding_window,
    # use the transformers default pattern of 6 for Gemma 3
    model_type = tc.get("model_type", "")
    if layer_types is None and sliding_window is not None and sliding_window_pattern is None:
        if "gemma3" in model_type:
            # Gemma 3 default: sliding_window_pattern=6
            # Source: transformers v5.1.0 configuration_gemma3.py:286
            layer_types = [
                "sliding_attention" if bool((i + 1) % 6)
                else "full_attention"
                for i in range(num_layers)
            ]

    # Count layer types
    if layer_types is not None:
        from collections import Counter
        counts = Counter(layer_types)
        n_sliding = counts.get("sliding_attention", 0)
        n_global = counts.get("full_attention", 0)
    else:
        n_sliding = 0
        n_global = num_layers

    return KVCacheProfile(
        num_sliding_layers=n_sliding,
        num_global_layers=n_global,
        num_kv_shared_layers=tc.get("num_kv_shared_layers", 0),
        sliding_window=sliding_window if isinstance(sliding_window, int) else None,
        head_dim=head_dim,
        global_head_dim=tc.get("global_head_dim"),
        num_kv_heads=num_kv_heads,
        num_global_kv_heads=tc.get("num_global_key_value_heads"),
        attention_k_eq_v=tc.get("attention_k_eq_v", False),
        triattn_budget=None,  # future: from preset kv_compression block
    )
```

### Key design decisions in the prototype

1. **`text_config` resolution:** Multimodal models (Gemma 3/3n/4) nest text
   model config under `text_config`. The prototype falls through to top-level
   config for single-modality models (Qwen, Llama).

2. **Gemma 3 pattern inference:** Gemma 3's config.json does NOT contain
   `layer_types` or `sliding_window_pattern`. The 5:1 sliding/global pattern is
   derived at model load time by the `Gemma3TextConfig` class using
   `sliding_window_pattern=6` (default). The fetcher must hardcode this knowledge
   for `model_type=gemma3_text`. Source: [transformers v5.1.0 configuration_gemma3.py:286-292](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/gemma3/configuration_gemma3.py).

3. **Graceful fallback:** If `_fetch_raw_config()` fails (gated repo, network
   error), the fetcher still produces a valid `ModelInfo` with the existing
   scalar `kv_cache_per_1k_tokens_mb` and `kv_cache_profile = None`.

---

## Step 5: Reference Values

### Model configurations (from config.json, fetched 2026-05-24)

| Field | Gemma 3n E4B | Gemma 3 27B | Qwen3 32B | Llama 3.3 70B |
|-------|-------------|-------------|-----------|---------------|
| Source | `google/gemma-3n-E4B-it` | `google/gemma-3-27b-it` | `Qwen/Qwen3-32B` | `unsloth/Llama-3.3-70B-Instruct` |
| `num_hidden_layers` | 35 | 62 | 64 | 80 |
| `num_attention_heads` | 8 | 32 | 64 | 64 |
| `num_key_value_heads` | 2 | 16 | 8 | 8 |
| `head_dim` | 256 | 128 | 128 | 128 |
| `hidden_size` | 2048 | 5376 | 5120 | 8192 |
| `sliding_window` | 512 | 1024 | None (null) | None |
| `sliding_window_pattern` | 5 (explicit `layer_types`) | 6 (inferred from transformers default) | N/A | N/A |
| Sliding layers | 28 | 52 | 0 | 0 |
| Global layers | 7 | 10 | 64 | 80 |
| `num_kv_shared_layers` | 15 | 0 | 0 | 0 |
| Unique sliding | 16 | 52 | 0 | 0 |
| Unique global | 4 | 10 | 64 | 80 |
| `attention_k_eq_v` | false | false | false | false |
| `model_type` | gemma3n_text | gemma3_text | qwen3 | llama |
| Config nesting | `text_config` | `text_config` | top-level | top-level |

Notes:
- Gemma 3n E4B: `layer_types` is explicit in config.json with 28 sliding + 7 full.
  `num_kv_shared_layers=15` means the last 15 layers reuse KV from an upstream layer.
  Actual distribution: 12 shared sliding + 3 shared global = 15 total.
  Source: [Botmonster Gemma 4 architecture](https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/).
- Gemma 3 27B: config.json has `sliding_window=1024` but NO `layer_types` field.
  The 5:1 pattern (52 sliding + 10 global) is derived from `sliding_window_pattern=6`
  which is a transformers default, not a config.json field.
  Source: [transformers v5.1.0 Gemma3TextConfig](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/gemma3/configuration_gemma3.py).
- Llama 3.3 70B: `meta-llama/Llama-3.3-70B-Instruct` is gated. Config fetched from
  `unsloth/Llama-3.3-70B-Instruct` mirror. Architecture fields are identical.

### KV per 1K tokens (linear formula, current gpumod)

| Model | Formula | MB/1K |
|-------|---------|-------|
| Gemma 3n E4B | `2 * 35 * 2 * 256 * 2 * 1000 / 1024^2` | 69 |
| Gemma 3 27B | `2 * 62 * 16 * 128 * 2 * 1000 / 1024^2` | 485 |
| Qwen3 32B | `2 * 64 * 8 * 128 * 2 * 1000 / 1024^2` | 250 |
| Llama 3.3 70B | `2 * 80 * 8 * 128 * 2 * 1000 / 1024^2` | 313 |

### Reference value table: Linear vs Compound KV cache (MB)

| Model | Context | Linear MB | Compound MB | Delta MB | Overestimate % |
|-------|---------|-----------|-------------|----------|----------------|
| Gemma 3n E4B | 8,000 | 546.9 | 78.5 | 468.4 | **85.6%** |
| Gemma 3n E4B | 32,000 | 2,187.5 | 266.0 | 1,921.5 | **87.8%** |
| Gemma 3n E4B | 128,000 | 8,750.0 | 1,016.0 | 7,734.0 | **88.4%** |
| Gemma 3 27B | 8,000 | 3,875.0 | 1,041.0 | 2,834.0 | **73.1%** |
| Gemma 3 27B | 32,000 | 15,500.0 | 2,916.0 | 12,584.0 | **81.2%** |
| Gemma 3 27B | 128,000 | 62,000.0 | 10,416.0 | 51,584.0 | **83.2%** |
| Qwen3 32B | 8,000 | 2,000.0 | 2,000.0 | 0.0 | 0.0% |
| Qwen3 32B | 32,000 | 8,000.0 | 8,000.0 | 0.0 | 0.0% |
| Qwen3 32B | 128,000 | 32,000.0 | 32,000.0 | 0.0 | 0.0% |
| Llama 3.3 70B | 8,000 | 2,500.0 | 2,500.0 | 0.0 | 0.0% |
| Llama 3.3 70B | 32,000 | 10,000.0 | 10,000.0 | 0.0 | 0.0% |
| Llama 3.3 70B | 128,000 | 40,000.0 | 40,000.0 | 0.0 | 0.0% |

### Effective KV MB per 1K tokens (varies with context for hybrid models)

| Model | Context | Linear/1K | Compound/1K |
|-------|---------|-----------|-------------|
| Gemma 3n E4B | 8,000 | 68.4 | 9.8 |
| Gemma 3n E4B | 32,000 | 68.4 | 8.3 |
| Gemma 3n E4B | 128,000 | 68.4 | 7.9 |
| Gemma 3 27B | 8,000 | 484.4 | 130.1 |
| Gemma 3 27B | 32,000 | 484.4 | 91.1 |
| Gemma 3 27B | 128,000 | 484.4 | 81.4 |
| Qwen3 32B | 8K-128K | 250.0 | 250.0 |
| Llama 3.3 70B | 8K-128K | 312.5 | 312.5 |

**Key insight:** For Gemma 3n E4B, the effective KV/1K drops from 68.4 (linear)
to 7.9-9.8 (compound) because most layers are sliding-window-bounded at 512
tokens, and 15 layers share KV entirely. The current formula reports this
model as needing 8,750 MB of KV cache at 128K context when it actually needs
only 1,016 MB -- a 7.7 GB overestimate that would prevent deploying it on
a 24 GB GPU when it actually fits.

### Backward compatibility verification

All 4 models tested: compound formula with `sliding_window=None` (infinity),
`triattn_budget=None` (infinity), `num_kv_shared_layers=0` produces values
identical (within 0.01 MB) to the current linear formula at ctx = 1K, 8K, 32K,
128K. **PASS.**

---

## Step 6: Preflight Update Sketch

### Current code (`vram_check.py:107-119`)

```python
# Strategy 2: Reduce context size
# KV cache roughly scales with context size
if ctx_size > 4096:
    # Halving context roughly halves KV cache
    suggested_ctx = ctx_size // 2
    kv_savings = overage // 2  # Conservative estimate
```

This heuristic is wrong for hybrid-attention models. For Gemma 3 27B at
ctx=128K, halving context to 64K does NOT halve KV cache:
- Linear formula: 62,000 MB -> 31,000 MB (halved, as claimed)
- Compound formula: 10,416 MB -> 5,416 MB (48% reduction, not 50%)

At ctx=8K, the difference is more dramatic:
- Linear: 3,875 -> 1,937.5 (halved)
- Compound: 1,041 -> 833 (20% reduction, NOT halved)

The sliding layers are already at their window cap (1024 tokens). Halving
context from 8K to 4K only reduces the global layers' contribution, which is
10 out of 62 layers.

### Proposed replacement (description, not code change)

The ctx-reduction suggestion in `VRAMSuggestion.for_llamacpp()` should be
replaced with a profile-aware calculation:

1. If the model has a `kv_cache_profile`:
   - Compute KV at current ctx and at proposed ctx using compound formula
   - Report actual savings: "Reduce ctx from 32K to 16K to save ~X MB"
   - If sliding window caps mean ctx reduction yields <10% savings, skip
     this suggestion entirely and suggest quantization instead

2. If the model has only scalar `kv_cache_per_1k_tokens_mb`:
   - Fall back to current heuristic (halving ctx roughly halves KV)

3. Add a new suggestion type for hybrid models: "KV cache is dominated by
   sliding-window layers capped at {W} tokens. Reducing context below {W}
   will have minimal KV impact. Consider a smaller quantization instead."

### Additional preflight check

For models with `kv_cache_profile`, the `VRAMCheck` should also:
- Warn if `context_size < sliding_window` (wasted attention capacity)
- Report the breakdown: "KV: {X} MB sliding + {Y} MB global = {Z} MB total"

---

## Step 7: Deliverables

### 1. Research report

This document: `docs/research/20260524_kv_estimation_redesign.md`

### 2. Proposed ModelInfo schema (additive, backward-compatible)

```python
class KVCacheProfile(BaseModel):
    """Structured KV cache data for layer-type-aware estimation."""

    model_config = ConfigDict(extra="forbid")

    num_sliding_layers: int = 0
    num_global_layers: int = 0
    num_kv_shared_layers: int = 0
    sliding_window: int | None = None
    head_dim: int = 128
    global_head_dim: int | None = None
    num_kv_heads: int = 1
    num_global_kv_heads: int | None = None
    attention_k_eq_v: bool = False
    triattn_budget: int | None = None
    kv_per_1k_at_inf: int | None = None   # backward-compat: linear-equivalent rate


class ModelInfo(BaseModel):
    # ... existing fields UNCHANGED ...
    kv_cache_per_1k_tokens_mb: int | None = None      # KEPT
    kv_cache_profile: KVCacheProfile | None = None     # NEW, optional
```

DB migration: `ALTER TABLE models ADD COLUMN kv_cache_profile TEXT` (JSON blob).

### 3. PoC demonstrating compound formula

File: `docs/research/poc/kv_estimation_compound.py`

Demonstrates:
- Compound formula implementation with all 6 architecture cases
- Backward-compatibility verification (compound with inf windows = linear)
- Reference value computation for all 4 models
- Detail breakdown showing per-layer-type contributions

Run: `uv run python docs/research/poc/kv_estimation_compound.py`

### 4. Reference-value table

See Step 5 above. Key finding: current formula overestimates KV cache by
73-88% for Gemma hybrid-attention models.

### 5. Persistence shape recommendation

**Option (a): Additive.** Keep `kv_cache_per_1k_tokens_mb`, add
`kv_cache_profile`. Migration cost: ~115 lines added, 0 existing lines changed,
0 preset files changed. See Step 3 for full rationale.

---

## Risks and Open Questions

1. **Gemma 3 pattern inference.** The 5:1 sliding/global pattern for Gemma 3 is
   NOT in config.json -- it's a transformers default (`sliding_window_pattern=6`).
   If Google changes this default in a future transformers release, the fetcher's
   hardcoded inference will produce wrong results. Mitigation: pin the pattern
   inference to known model_type values and log a warning for unknown types.

2. **`num_kv_shared_layers` distribution.** The proportional distribution
   (shared layers distributed across types based on layer ratio) was verified to
   match the actual Gemma 3n E4B distribution exactly. But this is one model --
   future models might distribute shared layers differently. Mitigation: when
   `layer_types` is available, count the actual shared layers by type using the
   "last N layers" rule. The proportional heuristic is the fallback.

3. **`attention_k_eq_v` impact.** None of the 4 reference models use K=V
   sharing. Gemma 4 models (26B-A4B, 31B) do. This halves KV storage but was
   not numerically verified in this PoC because the Gemma 4 models were not in
   scope. Follow-up ticket should verify.

4. **KV cache dtype.** The formula assumes fp16 (2 bytes per element). Some
   backends (llama.cpp with `--cache-type-k q8_0`) use quantized KV caches.
   The compound formula does not account for this. [unverified] whether this
   should be a profile field or a runtime parameter.

5. **TriAttention budget integration.** The `triattn_budget` field is included
   in the formula for forward compatibility but has no current producer. When
   TriAttention is integrated (see `docs/research/20260419_triattention_viability.md`),
   the budget will come from the preset `kv_compression` block, not from
   config.json. The formula already handles this case correctly.

---

## References

### Primary sources (config.json, fetched 2026-05-24)

- `google/gemma-3n-E4B-it` config.json `text_config` -- layer_types, sliding_window, num_kv_shared_layers
- `google/gemma-3-27b-it` config.json `text_config` -- sliding_window, head_dim, num_kv_heads
- `Qwen/Qwen3-32B` config.json -- dense architecture, sliding_window=null
- `unsloth/Llama-3.3-70B-Instruct` config.json -- dense architecture (mirror of gated meta-llama repo)

### Transformers source

- [Gemma3TextConfig sliding_window_pattern default](https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/gemma3/configuration_gemma3.py) -- `sliding_window_pattern=6` at line 286
- [Gemma3n HF docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3n) -- KV cache sharing, architecture overview
- [Gemma 3 HF docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3) -- sliding window pattern

### Architecture documentation

- [Botmonster: Gemma 4 Architecture](https://botmonster.com/posts/gemma-4-architecture-per-layer-embeddings-shared-kv-cache-dual-rope/) -- num_kv_shared_layers semantics
- [HF Blog: Welcome Gemma 3](https://huggingface.co/blog/gemma3) -- 5:1 sliding/global pattern
- [HF Blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4) -- dual RoPE, KV sharing, asymmetric head_dim

### Prior gpumod research

- `docs/research/20260419_triattention_viability.md` section A -- compound formula first proposal
- `src/gpumod/fetchers/huggingface.py:263-299` -- current linear formula
- `src/gpumod/registry.py:159-163` -- current VRAM estimation consumer

### huggingface_hub API

- [hf_hub_download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/download) -- file download with caching

---

## Follow-up Tasks

1. **Implement KVCacheProfile model and DB migration** -- add the new Pydantic
   model and `ALTER TABLE` migration. ~25 lines.
2. **Extend HuggingFaceFetcher** -- add `_fetch_raw_config()` and
   `_build_kv_cache_profile()` per Step 4 prototype. ~50 lines.
3. **Update registry.py estimation** -- prefer `kv_cache_profile` when
   available, fall back to scalar. ~20 lines.
4. **Update preflight ctx-reduction suggestion** -- profile-aware savings
   calculation per Step 6. ~20 lines.
5. **Verify Gemma 4 models** -- the `attention_k_eq_v`, `global_head_dim`, and
   `num_global_key_value_heads` fields need numerical verification on Gemma 4
   26B-A4B and 31B configs.
6. **KV cache dtype support** -- investigate whether `--cache-type-k q8_0`
   (llama.cpp) and FP8 KV cache (vLLM) should be represented in the profile.
