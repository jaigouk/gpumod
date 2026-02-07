# Nemotron-3-Nano-30B-A3B Spike Results

**Date:** 2026-02-07
**Ticket:** gpumod-bjb (Phase B: Spike)
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM)

## Model Overview

| Property | Value |
|---|---|
| Architecture | Hybrid Mamba2 + MoE (nemotron_h_moe) |
| Total Parameters | 30B |
| Active Parameters | ~3.5B (sparse MoE) |
| Max Context | 1M tokens |
| Reasoning | Built-in (toggle via chat_template_kwargs) |
| Tool Calling | Supported via Jinja template |

## Quantization Comparison (from HuggingFace API)

Registered via `gpumod model register --source huggingface --quant <name>`:

| Quant | File Size | Est. VRAM (1.1x) | Fits 24GB? |
|---|---|---|---|
| UD-Q4_K_XL | 21.3 GB | 23,954 MB | Yes (tight) |
| Q4_K_M | 22.9 GB | 25,780 MB | No |
| IQ4_NL | 16.9 GB | 19,049 MB | Yes |
| IQ4_XS | 16.9 GB | 19,038 MB | Yes |
| Q3_K_M | ~14 GB | ~15,400 MB | Yes |

**Finding:** Standard Q4_K_M is larger than UD-Q4_K_XL due to Unsloth's
dynamic bit allocation. Q4_K_M does NOT fit in 24 GB alongside embedding.

## VRAM Measurements (Live)

### UD-Q4_K_XL on RTX 4090

| Config | Context | Nemotron VRAM | Total VRAM | Free |
|---|---|---|---|---|
| Default (no --ctx-size, no --flash-attn) | model default | 20,008 MiB | 22,595 MiB | 1,487 MiB |
| Via router (--no-models-autoload) | lazy | ~0 MiB idle | 2,575 MiB | 21,507 MiB |

Baseline: vllm-embedding-code = 2,552 MiB

### gpumod Simulation

```
gpumod simulate mode nemotron --visual
```

```
Proposed: (22500 MB used)
[██████████████████████████████████████████████░░░░]
  Qwen3-Embedding-0.6B: 2500 MB
  Nemotron-3-Nano-30B-A3B: 20000 MB
  free: 2064 MB
```

## Inference Performance

| Metric | Value |
|---|---|
| Prompt processing | ~185-290 tok/s |
| Generation speed | ~96-100 tok/s |
| Time to first token | <200ms (warm) |
| Model load time | ~60s (cold load via router) |

## Reasoning Toggle

Nemotron supports per-request reasoning on/off via `chat_template_kwargs`:

**Reasoning ON (default):**
```json
{
  "model": "Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 200
}
```
Response includes `reasoning_content` + `content` (45 tokens for simple question).

**Reasoning OFF:**
```json
{
  "chat_template_kwargs": {"enable_thinking": false}
}
```
Response has only `content` (2 tokens for same question). Much faster for simple queries.

## Router Mode Compatibility

- Model loads/unloads via `POST /models/load` and `/models/unload`
- Supports `--no-models-autoload` (lazy loading, 0 VRAM idle)
- `--models-max 1` ensures single model at a time
- Works with `--jinja` for reasoning/tool calling templates
- Shares port 7070 with glm-code (router switches between them)

## Bugs Fixed During Spike

1. **GGUF magic constant** (src/gpumod/fetchers/gguf.py): Was `0x46475547`,
   should be `0x46554747`. Real GGUF files were rejected.
2. **Quant detection** (src/gpumod/fetchers/gguf.py): Missing `Q4_K_XL` and
   `Q8_K_XL` patterns from Unsloth's UD quants.
3. **HuggingFace GGUF support** (src/gpumod/fetchers/huggingface.py): Added
   ability to estimate VRAM from GGUF file sizes via HF API without downloading.
   Now supports `--quant` filter for picking specific quantization.

## Recommended Configuration

For RTX 4090 (24 GB) with vllm-embedding-code:

```yaml
# presets/llm/nemotron-3-nano.yaml
id: nemotron-3-nano
name: Nemotron-3-Nano-30B-A3B (llama.cpp)
driver: llamacpp
port: 7070
vram_mb: 20000
supports_sleep: true
sleep_mode: router
unit_vars:
  models_dir: $HOME/bin
  models_max: 1
  no_models_autoload: true
  jinja: true
```

### Router Preset INI (to add to glm-preset.ini)

```ini
[nemotron-3-nano]
model = $HOME/bin/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf
ctx-size = 8192
flash-attn = on
jinja = true
```

**Context size note:** Default context uses ~20 GB VRAM. For larger contexts,
consider KV cache quantization (`--cache-type-k q4_1 --cache-type-v q4_1`)
or mmap mode (`-ngl 0 --mmap`). These require further testing in Phase C.

## Phase C: Simulation Results

| Scenario | Total VRAM | Headroom | Fits? |
|---|---|---|---|
| Nemotron alone | 20,000 MB | 4,564 MB | Yes |
| Nemotron + embed-code (primary) | 22,500 MB | 2,064 MB | Yes |
| Nemotron + both embeddings | 27,500 MB | -2,936 MB | **No** |
| Current code mode (comparison) | 22,500 MB | 2,064 MB | Yes |

**Go/No-Go:** GO for Nemotron + embed-code mode. Same VRAM profile as current
code mode. Adding the general embedding (vllm-embedding, 5 GB) is not possible
alongside Nemotron -- would need sleep/wake or mmap mode for that scenario.

## Next Steps (Phase D)

1. Add nemotron-3-nano to `glm-preset.ini` with calibrated ctx-size
2. Test KV cache quantization for larger context windows
3. Measure mmap vs full GPU load trade-offs
4. Run `gpumod simulate` with context overrides to plan optimal config
5. Verify mode switching between code/nemotron modes
