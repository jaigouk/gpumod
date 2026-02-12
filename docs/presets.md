# Presets Reference

Presets are YAML files that define service configurations for repeatable
deployments. They are loaded during `gpumod init` and converted into
registered services.

## Preset directory structure

```
presets/
  llm/
    llama-3.1-8b.yaml
    mistral-7b.yaml
    qwen-2.5-72b-gguf.yaml
    devstral-small-2.yaml
  embedding/
    bge-large.yaml
    nomic-embed.yaml
```

## Preset YAML schema

Each preset file must conform to the `PresetConfig` schema:

```yaml
# Required fields
id: llama-3-1-8b              # Unique service identifier
name: Llama 3.1 8B Instruct   # Human-readable name
driver: vllm                   # Driver type: vllm, llamacpp, or fastapi
vram_mb: 8192                  # VRAM allocation in megabytes

# Optional fields
port: 8000                     # Service port number
context_size: 8192             # Context window size in tokens
kv_cache_per_1k: 32           # KV cache memory per 1000 tokens (MB)
model_id: meta-llama/Llama-3.1-8B-Instruct  # HuggingFace model ID
model_path: $HOME/models/model.gguf          # File path (env vars expanded)
health_endpoint: /health       # Health check endpoint (default: /health)
startup_timeout: 120           # Startup timeout in seconds (default: 60)
supports_sleep: true           # Whether the service supports sleep modes
sleep_mode: l1                 # Sleep mode: none, l1, l2, or router
unit_template: custom.j2       # Custom Jinja2 template name
unit_vars:                     # Variables passed to the systemd template
  gpu_mem_util: 0.9
  max_model_len: 8192
```

## Driver types

| Driver | Use Case | Template |
|--------|----------|----------|
| `vllm` | vLLM inference server | `vllm.service.j2` |
| `llamacpp` | llama.cpp server | `llamacpp.service.j2` |
| `fastapi` | Custom FastAPI server | `fastapi.service.j2` |
| `docker` | Docker container | N/A (uses Docker SDK) |

## Built-in presets

gpumod ships with example presets for common ML workloads:

| Preset | Driver | Model | VRAM |
|--------|--------|-------|------|
| `llama-3.1-8b` | vLLM | Llama 3.1 8B Instruct | 8 GB |
| `mistral-7b` | vLLM | Mistral 7B Instruct v0.3 | 6 GB |
| `devstral-small-2` | vLLM | Devstral Small 2505 | 15 GB |
| `qwen-2.5-72b-gguf` | llama.cpp | Qwen 2.5 72B Q4_K_M | 20 GB |
| `bge-large` | FastAPI | BGE Large EN v1.5 | 1 GB |
| `nomic-embed` | FastAPI | Nomic Embed Text v1.5 | 2 GB |

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

## Example: vLLM preset

```yaml
id: llama-3-1-8b
name: Llama 3.1 8B Instruct
driver: vllm
port: 8000
vram_mb: 8192
context_size: 8192
kv_cache_per_1k: 32
model_id: meta-llama/Llama-3.1-8B-Instruct
health_endpoint: /health
startup_timeout: 120
supports_sleep: true
sleep_mode: l1
unit_vars:
  gpu_mem_util: 0.9
  max_model_len: 8192
```

## Example: llama.cpp GGUF preset

```yaml
id: qwen-2-5-72b-gguf
name: Qwen 2.5 72B GGUF Q4_K_M
driver: llamacpp
port: 8080
vram_mb: 20480
context_size: 32768
kv_cache_per_1k: 128
model_id: Qwen/Qwen2.5-72B-Instruct-GGUF
model_path: $HOME/models/qwen2.5-72b-instruct-q4_k_m.gguf
health_endpoint: /health
startup_timeout: 180
supports_sleep: true
sleep_mode: l2
unit_vars:
  n_gpu_layers: 80
  threads: 8
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
- [Architecture](ARCHITECTURE.md) — System overview
