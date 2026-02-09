---
title: QA Testing
description: Manual QA procedure and results for gpumod mode switching, service lifecycle, and template rendering.
---

# QA Testing

## How to QA

QA validates that gpumod can switch modes, start/stop services, and serve
inference requests end-to-end on real hardware. Run through each step
sequentially -- every mode switch should cleanly stop the previous mode's
services before starting the new ones.

### Prerequisites

- NVIDIA GPU with `nvidia-smi` working
- systemd user session (`systemctl --user` functional)
- Unit files installed (`gpumod template install --yes`)
- Model weights downloaded for each preset being tested

### QA Sequence

1. **Blank mode** -- switch to `blank`, verify all services stopped.

2. **RAG mode** -- switch to `rag`, verify embedding service responds:
   ```bash
   gpumod mode switch rag
   curl -s http://127.0.0.1:8200/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"test","model":"Qwen/Qwen3-VL-Embedding-2B"}' \
     | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['data'][0]['embedding']), 'dims')"
   ```

3. **Code mode** -- switch to `code`, load a model via the llama.cpp router,
   verify chat completion:
   ```bash
   gpumod mode switch code
   curl -s -X POST http://127.0.0.1:7070/models/load \
     -H "Content-Type: application/json" \
     -d '{"model":"GLM-4.7-Flash-UD-Q4_K_XL"}'
   # wait for model to load
   curl -s http://127.0.0.1:7070/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"GLM-4.7-Flash-UD-Q4_K_XL","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
   ```

4. **Nemotron mode** -- switch to `nemotron`, load model, verify chat with
   reasoning output:
   ```bash
   gpumod mode switch nemotron
   curl -s -X POST http://127.0.0.1:7070/models/load \
     -H "Content-Type: application/json" \
     -d '{"model":"Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"}'
   # wait for model to load (~45s for 23GB GGUF)
   curl -s http://127.0.0.1:7070/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30}'
   ```

5. **Blank mode (cleanup)** -- switch back to `blank`, verify all services
   stopped:
   ```bash
   gpumod mode switch blank
   systemctl --user list-units --type=service --state=active | grep -E 'vllm|llama|glm|nemotron'
   ```

6. **Standalone service** -- start a single service outside any mode:
   ```bash
   gpumod service start devstral-small-2
   curl -s http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"mistralai/Devstral-Small-2505","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
   gpumod service stop devstral-small-2
   ```

### What to check

- Health endpoint responds before sending inference requests
- Mode switch stops previous mode's services (no orphan processes)
- `nvidia-smi` shows expected VRAM usage
- Service logs are clean (`journalctl --user -u <unit> --no-pager -n 20`)
