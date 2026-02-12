# Preset Modification Workflow

This guide documents the safe workflow for modifying gpumod presets, especially
for llama.cpp services where VRAM constraints are critical.

## Why This Matters

Blindly updating presets (e.g., `n_gpu_layers=50`) without validating VRAM
requirements causes OOM failures:

```
cudaMalloc failed: out of memory
```

The workflow below prevents these failures through a combination of automated
checks and manual validation.

---

## Preflight Checks

### Automated (run before service start)

gpumod's preflight system automatically validates these before starting a service:

| Check | What It Does | Failure Action |
|-------|--------------|----------------|
| **ModelFileCheck** | Verifies GGUF file exists at `unit_vars.models_dir/model_file` | Shows download URL and wget/curl commands |
| **DiskSpaceChecker** | Validates disk space before download (10% safety buffer) | Blocks download if insufficient |
| **TokenizerCheck** | Verifies tokenizer accessibility | Warns if tokenizer unavailable |

These checks run automatically via `PreflightRunner.default()` — no manual
action required.

### Manual (your responsibility)

VRAM validation is **not yet automated**. You must manually verify:

1. VRAM budget fits the model configuration
2. Configuration parameters (`n_gpu_layers`, `ctx_size`, etc.) are correct
3. MoE offload settings are appropriate (if applicable)

---

## Preset Modification Checklist

Follow this checklist when modifying any llama.cpp preset:

### Step 1: Check Current VRAM Budget

```bash
# MCP tool
gpumod gpu_status

# Or via nvidia-smi
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits
```

Note the **free VRAM** in MB. For RTX 4090: max is ~24000 MB.

### Step 2: Estimate VRAM Required

Use `list_gguf_files` to get VRAM estimates:

```bash
gpumod list_gguf_files --repo_id "unsloth/Qwen3-Coder-Next-GGUF"
```

This shows file sizes and estimated VRAM for each quant.

#### VRAM Calculation Formula

```
Total VRAM = Model Weights + KV Cache + Overhead

Model Weights ≈ GGUF file size × 1.1 (10% for buffers)

KV Cache = ctx_size × n_layers × head_dim × 2 × dtype_size × n_gpu_layers_ratio
         ≈ ctx_size × 2MB per 1K tokens (for 70B models with f16 cache)
         ≈ ctx_size × 0.5MB per 1K tokens (for 70B models with q8_0 cache)

Overhead ≈ 500MB (llama.cpp runtime, CUDA context)
```

### Step 3: Calculate for Common Scenarios

#### Dense Models (e.g., Qwen2.5-72B, Llama-70B)

| Quant | File Size | Model VRAM | 8K ctx (f16) | Total |
|-------|-----------|------------|--------------|-------|
| Q4_K_M | ~42 GB | ~46 GB | ~16 GB | ~63 GB |
| Q4_K_S | ~40 GB | ~44 GB | ~16 GB | ~61 GB |
| Q3_K_M | ~35 GB | ~39 GB | ~16 GB | ~56 GB |

**Partial offload** reduces VRAM proportionally:
- `n_gpu_layers=40` of 80 total → 50% in VRAM → ~23 GB + KV cache

#### MoE Models (e.g., Qwen3-Coder-Next 80B)

MoE models have additional offload controls:

```yaml
unit_vars:
  n_gpu_layers: 45   # Layers in VRAM
  n_cpu_moe: 512     # MoE experts on CPU
  ctx_size: 8192     # Context window
```

| Quant | File Size | Full VRAM | 45 layers | With n_cpu_moe=512 |
|-------|-----------|-----------|-----------|---------------------|
| Q2_K_XL | ~29 GB | ~32 GB | ~18 GB | ~22 GB |
| Q4_K_M | ~46 GB | ~51 GB | ~28 GB | ~24 GB (tight!) |

**MoE VRAM formula:**
```
MoE VRAM ≈ Base VRAM × (n_gpu_layers / total_layers) × (1 - cpu_moe_ratio)
```

### Step 4: Validate Fit

```
Required VRAM < Available VRAM - 1GB buffer
```

**Example:**
- Available: 24000 MB
- Required: 22000 MB
- Buffer: 1000 MB
- Check: 22000 < 24000 - 1000 = 23000 ✅

If it doesn't fit:
1. Reduce `n_gpu_layers`
2. Reduce `ctx_size`
3. Use a smaller quantization
4. Increase `n_cpu_moe` (MoE only)

### Step 5: Test Incrementally

Don't jump to maximum settings. Start conservative:

```yaml
# Start here
n_gpu_layers: 30
ctx_size: 4096

# Then increase gradually
n_gpu_layers: 40
ctx_size: 8192

# Monitor with
watch -n 1 nvidia-smi
```

### Step 6: Document Your Assumptions

Add comments explaining configuration choices:

```yaml
unit_vars:
  # Q2_K_XL (~29.5GB) - fits in 24GB VRAM + 32GB RAM with swap
  default_model: Qwen3-Coder-Next-UD-Q2_K_XL
  # Conservative: ~22GB VRAM, ~7.5GB in RAM. Tune up if stable.
  n_gpu_layers: 45
  n_cpu_moe: 512
  # Start with 8K context; increase if VRAM allows
  ctx_size: 8192
```

---

## Quick Reference: VRAM by Quant Type

| Quant | Bits/Weight | Relative Size | Quality | Use Case |
|-------|-------------|---------------|---------|----------|
| Q8_0 | 8.0 | 100% | Best | Have VRAM to spare |
| Q6_K | 6.5 | ~81% | Excellent | Recommended default |
| Q5_K_M | 5.5 | ~69% | Very Good | Good balance |
| Q4_K_M | 4.5 | ~56% | Good | Popular choice |
| Q4_K_S | 4.5 | ~54% | Good | Smaller Q4 |
| Q3_K_M | 3.4 | ~44% | OK | VRAM constrained |
| Q2_K | 2.6 | ~34% | Degraded | Extreme constraint |
| IQ2_XXS | 2.1 | ~26% | Degraded | Last resort |

**Rule of thumb:** Each quant step down saves ~15-20% VRAM.

---

## Troubleshooting

### OOM on Startup

```
cudaMalloc failed: out of memory
```

1. Check actual VRAM with `nvidia-smi`
2. Reduce `n_gpu_layers` by 10
3. Reduce `ctx_size` to 4096
4. Try smaller quantization

### Model Not Loading

```
Model file not found
```

Preflight should catch this. If you see it:
1. Check `unit_vars.models_dir` path
2. Check `unit_vars.model_file` filename
3. Download using the wget command from preflight output

### Slow Inference

If inference is slow but VRAM isn't full:
1. Check `n_gpu_layers` — more layers in VRAM = faster
2. Check `threads` — match to CPU cores
3. Check for swap usage (`free -h`)

---

## Related

- [Presets Reference](presets.md) — YAML schema and examples
- [Architecture](ARCHITECTURE.md) — System overview
- gpumod-89z (planned): Automated VRAM validation preflight check
