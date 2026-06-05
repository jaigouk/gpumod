# Presets Reference

Presets are YAML files that define service configurations for repeatable
deployments. They are loaded into the database via `gpumod init` or
`gpumod preset sync`, and rendered into systemd unit files by
`gpumod template install`.

## Preset directory structure

```
presets/
  llm/                         # 40+ LLM presets (Gemma 4, Qwen 3.5/3.6, GLM, Llama, etc.)
    gemma4-26b-a4b-q4.yaml         # single-slot vision-enabled
    gemma4-26b-a4b-q4-multi.yaml   # 3-slot text-only (gpumod-8xaq)
    qwen35-27b-q4-multi.yaml
    qwen3-coder-multi.yaml
    ...
  vllm/                        # vLLM servers (chat, embed, ASR, TTS, etc.)
  fastapi/                     # FastAPI servers (embeddings, custom backends)
```

## Preset YAML schema

Each preset file must conform to the `PresetConfig` Pydantic schema in
`src/gpumod/models.py`:

```yaml
# Required fields
id: gemma4-26b-a4b-q4-multi           # unique service identifier
name: Gemma 4 26B-A4B Multi-Agent     # human-readable name
driver: llamacpp                       # vllm | llamacpp | fastapi | docker
vram_mb: 20000                         # additive VRAM budget in MB

# Optional fields
port: 7109                             # service port
context_size: 131072                   # context window (tokens)
kv_cache_per_1k: 8                     # KV memory per 1K tokens (informational)
model_id: unsloth/gemma-4-26B-A4B-it-GGUF
model_path: $HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf   # $HOME expanded at install
health_endpoint: /health
startup_timeout: 240
supports_sleep: false                  # gpumod's sleep/wake support
sleep_mode: none                       # none | l1 | l2 | router
preflight_required: true               # run gpumod preflight before ExecStart
unit_template: custom.j2               # override the default driver template
unit_vars:                             # variables passed to the Jinja2 template
  model_path: $HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
  context_size: 393216                 # multi-slot: per_slot × parallel
  n_gpu_layers: -1
  jinja: true
  flash_attn: true
  extra_args: "--parallel 3 --cont-batching ..."   # raw driver-specific flags
```

`unit_vars.extra_args` is the most flexible field — it's appended verbatim to
the driver's `ExecStart` so any driver-specific flag works without changing
the template. See the [Multi-slot patterns](#multi-slot-patterns) section.

## Driver types

| Driver | Use Case | Template | Notes |
|--------|----------|----------|-------|
| `vllm` | vLLM inference server | `vllm.service.j2` | OpenAI-compatible, sleep_mode l1/l2 supported |
| `llamacpp` | llama.cpp server | `llamacpp.service.j2` | Sets `GGML_CUDA_NO_PINNED=1` by default (see [Defense flags](#defense-flags)) |
| `fastapi` | Custom FastAPI server | `fastapi.service.j2` | For embedding / custom backends |
| `docker` | Docker container | N/A — uses Docker SDK | Sandboxed; no `--privileged`, no host network |

## Built-in presets

gpumod ships 40+ presets in `presets/llm/`, `presets/vllm/`, and
`presets/fastapi/`. The set below is representative; run
`gpumod service list` for the full current catalog.

### LLM single-slot (one agent, full per-request bandwidth)

| Preset | Driver | Model | VRAM | Notes |
|--------|--------|-------|------|-------|
| `gemma4-26b-a4b-q4` | llama.cpp | Gemma 4 26B-A4B UD-IQ4_XS | 20 GB | Multimodal (vision via `--mmproj`) |
| `gemma4-12b-q4` / `-q5` / `-q8` | llama.cpp | Gemma 4 12B IT | 14-17 GB | Quality ladder |
| `qwen35-27b-q4` / `-q3` | llama.cpp | Qwen 3.5 27B dense | 18-20 GB | Dense (all params active) |
| `qwen35-35b-a3b-*` | llama.cpp | Qwen 3.5 35B-A3B MoE | 18-22 GB | Several imatrix variants |
| `llama-3.1-8b` | vLLM | Llama 3.1 8B Instruct | 8 GB | Small reference |
| `mistral-7b` | vLLM | Mistral 7B v0.3 | 6 GB | Small reference |

### LLM multi-slot (cont-batching, multi-agent)

| Preset | Driver | Backbone | Slots × Ctx | VRAM | Notes |
|--------|--------|----------|-------------|------|-------|
| `gemma4-26b-a4b-q4-multi` | llama.cpp | Gemma 4 26B-A4B | 3 × 128K | 20 GB | **Production hermes-agent / code mode** (gpumod-8xaq). 132 TPS aggregate. |
| `qwen35-27b-q4-multi` | llama.cpp | Qwen 3.5 27B dense | 3 × 40K | 18 GB | |
| `qwen35-35b-a3b-q4-multi` | llama.cpp | Qwen 3.5 35B-A3B MoE | 3 × 32K | 22 GB | |
| `qwen3-coder-multi` / `-multi-p3` | llama.cpp | Qwen 3 Coder | varies | 18 GB | |
| `gpt-oss-20b-multi` | llama.cpp | GPT-OSS 20B | 3 × varies | 18 GB | |

### Embedding / utility

| Preset | Driver | Model | VRAM |
|--------|--------|-------|------|
| `vllm-embedding-code` | vLLM | Qwen3-Embedding-0.6B | 2.5 GB |
| `vllm-embedding` | vLLM | Qwen3-VL-Embedding | 5 GB |
| `bge-large` | FastAPI | BGE Large EN v1.5 | 1 GB |
| `vllm-reranker` | vLLM | Qwen3-VL-Reranker | 6 GB |

## Creating custom presets

1. Create a YAML file following the schema above.
2. Place it in a directory (e.g., `~/gpumod-presets/llm/my-model.yaml`).
3. Initialize with the custom directory:

```bash
gpumod init --preset-dir ~/gpumod-presets
```

Or set the environment variable:

```bash
export GPUMOD_PRESETS_DIR=~/gpumod-presets
gpumod init
```

Re-sync after editing existing presets:

```bash
gpumod preset sync
gpumod template install <id> --yes    # re-render the systemd unit
```

## Multi-slot patterns

llama.cpp supports concurrent generation via `--parallel N --cont-batching`
— N independent KV cache slots, scheduler packs incoming requests across
them. From the operator's perspective each multi-slot preset is still **one
service / one systemd unit**; the multi-agent fan-out happens inside
llama-server.

Naming convention: append `-multi` (or `-multi-pN` for an explicit slot
count) to the preset id.

### Example — `presets/llm/gemma4-26b-a4b-q4-multi.yaml`

```yaml
id: gemma4-26b-a4b-q4-multi
name: Gemma 4 26B-A4B IT UD-IQ4_XS Multi-Agent (N=3, 128K per slot)
driver: llamacpp
port: 7109
vram_mb: 20000
context_size: 131072            # per-slot ctx (informational; the unit uses total)
kv_cache_per_1k: 8
model_id: unsloth/gemma-4-26B-A4B-it-GGUF
model_path: $HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
health_endpoint: /health
startup_timeout: 240
supports_sleep: false
sleep_mode: none
preflight_required: true
unit_vars:
  model_path: $HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
  context_size: 393216         # total context (per_slot × parallel)
  n_gpu_layers: -1
  jinja: true
  flash_attn: true
  extra_args: >-
    --parallel 3 --cont-batching --threads 16
    --cache-type-k q8_0 --cache-type-v q8_0
    --slot-save-path $HOME/.cache/llama-slots
    --temp 1.0 --top-p 0.95 --top-k 64
    --chat-template-kwargs '{"enable_thinking":true}'
```

Key knobs:

- `--parallel N` — number of slots (N=3 is the throughput knee on Gemma 4 26B-A4B; see [research](../research/20260604_multi_agent_hermes_capacity/README.md))
- `--cont-batching` — interleave decode steps across slots
- `--cache-type-k/v q8_0` — quantize KV cache (~8 MiB/1K per slot on Gemma 4 sliding-window layers)
- `--slot-save-path <dir>` — persist per-slot KV to disk for "agent disconnect/reconnect with memory" (see [Defense flags](#defense-flags))
- `--chat-template-kwargs '{"enable_thinking":true}'` — required for Gemma 4 / Qwen 3.5+ reasoning mode

**Empirical throughput curve** for Gemma 4 26B-A4B at N=3 (text-only, vllm-embedding-code co-tenant; full table in the [gpumod-8xaq research README](../research/20260604_multi_agent_hermes_capacity/README.md)):

| per-slot ctx | aggregate TPS | per-call TPS | verdict |
|---:|---:|---:|---|
| 64K | 142 | 70 | narrow throughput max |
| **128K** | **132** | **69** | **sweet spot — production target** |
| 200K | 79 | 29 | -45% TPS, 3× Dev latency |
| 256K | 10 | — | **collapsed** — attention compute scales with allocated slot ctx |

The cliff between 200K and 256K is the reason `gemma4-26b-a4b-q4-multi` ships
at 128K per slot, not the model's native 256K. Do not raise ctx above 128K
per slot without re-benching.

## Defense flags

### `GGML_CUDA_NO_PINNED=1` (default for all `llamacpp` services)

Set unconditionally in `src/gpumod/templates/systemd/llamacpp.service.j2`.
Bypasses `cudaMallocHost`, eliminating the page-fragmentation freeze class
where NVIDIA's driver hangs waiting for contiguous high-order pages and no
OOM signal fires (gpumod-x7rv root-cause; gpumod-56md fix).

Cost: ~0.3% TPS regression measured 2026-05-26. **Do not remove without a
fresh benchmark on the lower RAM floor.**

### `--slot-save-path <dir>` (multi-slot persistence)

Enables the REST endpoints `POST /slots/{id}?action=save` and
`?action=restore` so clients can persist per-slot KV cache to disk between
disconnects. Validated on Gemma 4 26B-A4B in gpumod-8viu: save 3.8 ms /
restore 2.3 ms for 41 tokens of state; ~120 KB/token on disk (bf16
full-precision). Zero VRAM cost from enabling the flag.

Requires the target directory to exist before service start:

```bash
mkdir -p ~/.cache/llama-slots
```

### `--swa-full` — **DO NOT add to multi-slot Gemma 4 presets**

The llama.cpp [persistent-KV tutorial](https://github.com/ggml-org/llama.cpp/discussions/20572)
recommends `--swa-full` for sliding-window-attention models. On Gemma 4 at
ctx=393216 (3 × 128K) this forces every layer to allocate full-ctx KV
instead of the 1024-token sliding window, blowing past the 24 GiB VRAM
ceiling and triggering the cudaHostAlloc-class host freeze (gpumod-8viu
v1 incident, 2026-06-05 — hard reboot required).

Slot save/restore on Gemma 4 works correctly **without** `--swa-full`.

### Preflight RAMCheck

When `preflight_required: true`, gpumod's `ExecStartPre` runs
`gpumod preflight all` which refuses to boot if MemAvailable is below
`model_size × 1.1 + 1024 MB`. Empirically calibrated; don't relax without
a fresh `GGML_CUDA_NO_PINNED` benchmark on the new floor.

## Vision-enabled vs text-only

llama.cpp's `--mmproj <path>` loads a vision encoder GGUF alongside the
language model so image inputs work through the OpenAI-compatible chat
endpoint.

| Variant | `--mmproj` | per-slot ctx ceiling | When to pick |
|---|---|---|---|
| `gemma4-26b-a4b-q4` (single-slot) | yes | 128K | Image review, chart reading, single-user vision tasks |
| `gemma4-26b-a4b-q4-multi` (3-slot) | **no** | 128K | Multi-agent text workflows; ~1.1 GiB VRAM headroom budgeted to per-slot KV instead of mmproj |

Swap modes by editing the relevant `modes/*.yaml` to reference the other
preset, run `gpumod mode sync`, then `gpumod mode switch`.

## Example: vLLM preset

```yaml
id: vllm-embedding-code
name: Qwen3-Embedding-0.6B (Code)
driver: vllm
port: 8210
vram_mb: 2500
model_id: Qwen/Qwen3-Embedding-0.6B
health_endpoint: /health
startup_timeout: 120
supports_sleep: true
sleep_mode: l1
unit_vars:
  gpu_memory_utilization: 0.10
  max_model_len: 4096
  enforce_eager: true
  served_model_name: Qwen/Qwen3-Embedding-0.6B
```

## Example: FastAPI embedding preset

```yaml
id: bge-large
name: BGE Large EN v1.5
driver: fastapi
port: 9200
vram_mb: 1024
model_id: BAAI/bge-large-en-v1.5
health_endpoint: /health
startup_timeout: 60
supports_sleep: false
sleep_mode: none
unit_vars:
  app_module: embedding_server:app
  working_dir: /opt/embedding
```

## Example: Docker container preset

```yaml
id: ollama
name: Ollama LLM Server
driver: docker
port: 11434
vram_mb: 8192
health_endpoint: /api/tags
startup_timeout: 120
extra_config:
  image: ollama/ollama:latest
  ports:
    - "11434:11434"
  runtime: nvidia
  volumes:
    ~/ollama-models: /root/.ollama
  environment:
    OLLAMA_MODELS: /root/.ollama
    OLLAMA_NUM_PARALLEL: "2"
```

Docker presets use `extra_config` for container settings (image, ports,
environment variables, volumes). The Docker driver enforces security
controls: no `--privileged`, no host network, no unsafe volume mounts,
and environment variable sanitization.

## See Also

- [Preset Modification Workflow](presets-workflow.md) — VRAM validation checklist
- [Architecture](../architecture/index.md) — system overview
- [gpumod-8xaq research](../research/20260604_multi_agent_hermes_capacity/README.md) — multi-slot capacity findings
- [gpumod-8viu research](../research/20260605_slot_persistence/README.md) — slot save/restore validation
- [gpumod-x7rv findings](../research/20260525_oom_protection_findings/FINDINGS.md) — cudaHostAlloc freeze root cause
