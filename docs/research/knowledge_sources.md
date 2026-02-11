# Knowledge Sources for RLM Consult Tool

This document catalogs validated knowledge sources for the RLM consult tool,
organized by tier based on reliability and update frequency.

## Tier 1: Core Tool Documentation (Live Fetch)

Official documentation that should be fetched live for up-to-date information.

| Source | URL | Content |
|--------|-----|---------|
| llama.cpp Server | https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md | Server flags, API endpoints, sampling params |
| llama.cpp Quantize | https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md | Quantization types, imatrix usage |
| vLLM Engine Args | https://docs.vllm.ai/en/stable/configuration/engine_args/ | Engine configuration, parallelism |
| vLLM KV Cache | https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | FP8 KV cache quantization |
| vLLM FP8 | https://docs.vllm.ai/en/latest/features/quantization/fp8/ | FP8 W8A8 quantization |

### Fetch Strategy
- TTL: 24 hours (docs update infrequently)
- Fallback: Return cached if fetch fails
- Size: ~50-100KB per doc

## Tier 2: Curated Guides (Cached, Weekly Refresh)

High-quality guides from trusted sources, cached with periodic refresh.

| Source | URL | Content |
|--------|-----|---------|
| MoE Offload Guide | https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide | CPU+GPU hybrid for MoE models |
| Qwen llama.cpp | https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html | Qwen-specific quantization |
| GGUF Tensor Schemes | https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes | Q4_K_M, Q5_K_S, IQ types |
| Unsloth Docs | https://unsloth.ai/docs | Fine-tuning, GGUF export, tutorials |
| Unsloth Dynamic 4-bit | https://unsloth.ai/blog/dynamic-4bit | Dynamic quantization benefits |
| Unsloth Blog | https://unsloth.ai/blog | Model guides, training techniques |

### Key MoE Offload Flags (from guide)
```bash
-ngl 999                    # All layers to GPU
-ot "exps=CPU"              # Offload routed experts to CPU
--n-cpu-moe 31              # Number of expert layers on CPU
-b 4096 -ub 4096            # Batch sizes for MoE
```

### Fetch Strategy
- TTL: 7 days
- Store: Local cache with ETag validation
- Size: ~20-50KB per guide

### Unsloth Dynamic 2.0 Quantization
Key benefit: Selectively keeps critical parameters unquantized for better accuracy.

- Standard 4-bit often breaks small models (e.g., Qwen2 Vision 2B)
- Dynamic 4-bit adds only ~450MB but recovers accuracy
- Uses <10% additional VRAM vs standard BitsAndBytes 4-bit

### Popular Unsloth GGUF Models
| Model | Size | Downloads | Notes |
|-------|------|-----------|-------|
| Qwen3-Coder-Next-GGUF | 80B | 219k | Latest coding model |
| GLM-4.7-Flash-GGUF | 30B | 358k | Strong agentic/coding MoE |
| Kimi-K2.5-GGUF | 1T | 35k | Largest available |
| Nemotron-3-Nano-30B-A3B-GGUF | 32B | 47k | Efficient MoE |
| DeepSeek-V3.2-GGUF | 671B | 12k | Largest reasoning |

## Tier 3: Dynamic Data (API Fetch)

Data fetched from APIs that changes frequently.

| Source | API | Content |
|--------|-----|---------|
| HuggingFace config.json | `https://huggingface.co/{repo}/raw/main/config.json` | Model architecture, params |
| Unsloth Models | HF API: `unsloth` org | Available GGUF quants |
| Model README | `https://huggingface.co/{repo}/raw/main/README.md` | Recommended settings |

### Fetch Strategy
- TTL: 1 hour for config.json
- TTL: 6 hours for model listings
- Cache key: repo_id + file

## Tier 4: Community Knowledge (Curated, Manual Update)

Valuable community insights that require manual curation.

### VRAM Estimation Rules
Source: Community consensus + calculators

| Model Size | Q4_K_M | Q5_K_M | Q8_0 |
|------------|--------|--------|------|
| 7B | ~5GB | ~6GB | ~8GB |
| 13B | ~8GB | ~10GB | ~14GB |
| 32B | ~20GB | ~24GB | ~34GB |
| 70B | ~42GB | ~50GB | ~72GB |

Formula: `VRAM = params_B * bytes_per_weight + kv_cache + overhead`
- Q4_K_M: ~0.56 bytes/weight
- Q5_K_M: ~0.69 bytes/weight
- Q8_0: ~1.0 bytes/weight
- KV cache: `2 * layers * kv_heads * head_dim * context * 2 bytes`

### Quantization Recommendations (2026)
Source: [GGUF Quality vs Speed Analysis](https://dasroot.net/posts/2026/02/gguf-quantization-quality-speed-consumer-gpus/)

| VRAM | Recommended Quant | Notes |
|------|-------------------|-------|
| 8GB | Q4_K_M | Best balance for consumer GPUs |
| 12GB | Q5_K_M | Better quality, fits most 7B-13B |
| 16GB | Q6_K | Near-lossless for 7B-13B |
| 24GB | Q8_0 or Q5_K_M (32B) | Full precision small, or quant large |
| 48GB | Q8_0 (32B) or Q4_K_M (70B) | High quality 32B or workable 70B |

### Known Model Quirks
Manually curated from community reports:

| Model | Quirk | Solution |
|-------|-------|----------|
| GLM-4.7B | RoPE scaling issues | Use `--rope-scaling linear` |
| Qwen3 | `/think` mode token budget | Set `--max-tokens` appropriately |
| DeepSeek MoE | High CPU expert count | Use `-ot "exps=CPU"` with high `-b` |
| Llama 3.1 | Large KV cache | Consider FP8 KV or reduce context |

## Priority Order for RLM Consult

When answering queries, consult sources in this order:

1. **Model config.json** - Get actual architecture (params, layers, context)
2. **Driver docs** - Get available flags for llamacpp/vllm
3. **Quantization guide** - Recommend appropriate quant
4. **VRAM rules** - Estimate if model fits
5. **Community quirks** - Check for known issues

## Source Validation Status

| Source | Last Verified | Status |
|--------|---------------|--------|
| llama.cpp server README | 2026-02-11 | Valid |
| vLLM engine args | 2026-02-11 | Valid (URL updated to /configuration/) |
| MoE Offload Guide | 2026-02-11 | Valid |
| GGUF Tensor Schemes | 2026-02-11 | Valid |
| vLLM KV Cache | 2026-02-11 | Valid |
| Unsloth Docs | 2026-02-11 | Valid |
| Unsloth Dynamic 4-bit | 2026-02-11 | Valid |
| Unsloth HF Models | 2026-02-11 | Valid |

## References

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Hub](https://huggingface.co/)
- [Unsloth Documentation](https://unsloth.ai/docs)
- [Unsloth HF Organization](https://huggingface.co/unsloth)
- [NyxKrage VRAM Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)
