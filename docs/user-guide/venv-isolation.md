# Per-driver venv isolation

For services backed by Python-based drivers (vllm, fastapi/uvicorn), the systemd unit's `ExecStart` invokes a driver binary. By default, gpumod auto-detects that binary via `shutil.which` and uses whatever your shell PATH finds (typically a global interpreter).

A shared global interpreter works fine until a transitive dependency drifts. For example: vllm pins one version range of `transformers`, but another tool installed later upgrades `huggingface-hub`, which transformers refuses to import with — and the next time the vllm service restarts it crashloops. Repeated systemd `Restart=on-failure` cycles over an idle window can fragment the kernel's page allocator badly enough to trigger a silent hang on the next large CUDA-pinned allocation.

The remedy is to **isolate each Python-backed driver in its own venv** and point the service's `ExecStart` at that venv's binary.

## When you need this

You don't need venv isolation if:

- You only run llamacpp services (the llama-server binary is statically linked from `~/bin/llama.cpp/build` and not affected)
- Your machine runs one Python project (no cross-tool dep drift)

You **do** need venv isolation if any of these are true:

- You install multiple Python tools on the same machine that share an interpreter (LLM toolkits, image-gen frameworks, training stacks, MCP servers, etc.)
- An upgrade to one tool has ever broken another via transitive dep conflict
- A vllm service crashloops with `ImportError`, `AttributeError`, or version-range errors at startup

## How to set it up

1. **Create a dedicated venv** for the driver. `uv` is the recommended tool because it's fast and integrates with installed Python interpreters; any venv tool works.

   ```bash
   uv venv ~/.venvs/vllm --python 3.13
   VIRTUAL_ENV=~/.venvs/vllm uv pip install 'vllm==0.11.0'
   # If vllm pulls in conflicting versions of transformers/hf-hub, pin explicitly:
   VIRTUAL_ENV=~/.venvs/vllm uv pip install \
       'transformers>=4.55.2,<5.0' \
       'huggingface-hub<1.0'
   ```

2. **Point the service preset at the venv's binary** via `unit_vars.vllm_bin`. Edit your service preset YAML (e.g. `presets/llm/vllm-embedding-code.yaml`):

   ```yaml
   id: vllm-embedding-code
   driver: vllm
   port: 8210
   vram_mb: 2500
   model_id: Qwen/Qwen3-Embedding-0.6B
   extra_config:
     unit_vars:
       vllm_bin: /home/operator/.venvs/vllm/bin/vllm     # ← per-service override
       gpu_mem_util: 0.085
       max_model_len: 4096
       runner: pooling
       dtype: float16
   ```

3. **Re-render and re-install the unit**:

   ```bash
   gpumod template install vllm-embedding-code   # or whatever your operator workflow is
   systemctl --user daemon-reload
   systemctl --user restart vllm-embedding-code
   ```

4. **Verify the service comes up cleanly**:

   ```bash
   curl -sf http://127.0.0.1:8210/health   # should return 200 within startup_timeout
   ```

## Resolution order

The vllm template resolves the `ExecStart` binary path in this order:

1. **Per-service**: `unit_vars.vllm_bin` (in the service preset)
2. **Global**: `settings.vllm_bin` (set via `gpumod settings set vllm_bin <path>` or auto-detected via `shutil.which`)
3. **Fallback**: bare `vllm` (resolved by systemd against PATH at unit start time)

This means you can mix-and-match: most services use the global default, while a specific service that needs a different version range overrides via `unit_vars.vllm_bin`.

## Why not let the global env handle it

If you've ever tried to "just upgrade transformers" globally to satisfy vllm:

- Other tools you've installed (sentence-transformers, unsloth, custom training scripts) typically pin transformers `<5.0`
- Other tools that DO want `huggingface-hub >= 1.x` (image-gen frameworks, MCP servers) are blocked if you downgrade

The two camps are incompatible. Per-driver venvs sidestep the conflict instead of trying to resolve it.

## Future work

- The same pattern will apply to other Python-backed drivers (e.g. fastapi-based ASR/TTS services). Each gets its own venv when needed.
- A future `gpumod doctor venv` subcommand will validate the venv against compatibility ranges declared in the preset — see the project's open issues for status.
