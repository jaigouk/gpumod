# vLLM Sleep Mode — Research Spike

> Date: 2026-02-07
> Author: gpumod agent
> Source: vLLM docs (Context7), systemd unit inspection, driver code review

## Problem

`gpumod mode switch` currently uses full `systemctl stop` / `systemctl start` for
every service transition. This causes:

- **30-130 second mode switches** (cold start: CUDA init, model load from disk, warmup)
- **Unnecessary VRAM churn** — models are fully unloaded then reloaded
- **Wasted CPU/IO** — model weights re-read from disk/HF cache every time

## vLLM Sleep Mode Overview

vLLM supports two sleep levels that preserve the running process while freeing GPU memory.

### Sleep Levels

| Level | What happens | Wake time | VRAM freed |
|-------|-------------|-----------|------------|
| **L1** | Offloads weights to CPU RAM, discards KV cache | **~instant** (weights still in RAM) | Partial (KV cache freed) |
| **L2** | Discards both weights and KV cache from GPU | **~2-5s** (re-upload from CPU RAM) | Full model VRAM freed |

### Server Requirements

```bash
# Both flags required:
VLLM_SERVER_DEV_MODE=1 vllm serve <model> --enable-sleep-mode
```

- `VLLM_SERVER_DEV_MODE=1` — enables `/sleep`, `/wake_up`, `/is_sleeping` endpoints
- `--enable-sleep-mode` — enables the sleep mode feature in the engine
- **WARNING**: Dev mode endpoints should not be exposed to end users

### HTTP API

```bash
# Put model to sleep (L1: offload weights to CPU)
curl -X POST 'http://localhost:8000/sleep?level=1'

# Put model to deep sleep (L2: discard weights + KV cache)
curl -X POST 'http://localhost:8000/sleep?level=2'

# Wake up (restore all)
curl -X POST 'http://localhost:8000/wake_up'

# Wake up selectively (L2 only)
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'

# Check sleep state
curl -X GET 'http://localhost:8000/is_sleeping'
# → {"is_sleeping": true}
```

### L2 Full Wake Sequence

For L2 sleep, the recommended wake sequence avoids OOM:

```bash
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
curl -X POST 'http://localhost:8000/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights"}'
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

## Current System State

### Services with sleep mode enabled in systemd units

| Service | Sleep Level | `VLLM_SERVER_DEV_MODE` | `--enable-sleep-mode` |
|---------|------------|------------------------|----------------------|
| vllm-chat | L1 | YES | YES |
| vllm-hyde | L2 | YES | YES |
| vllm-reranker | L2 | YES | YES |
| vllm-embedding | none | no | no |
| vllm-embedding-code | none | no | no |
| vllm-tts | none | no | no |
| qwen3-asr | L1 (lazy) | N/A (FastAPI) | N/A |
| glm-code | router | N/A (llama.cpp) | N/A |

### Existing gpumod Driver Support

The vLLM and llama.cpp drivers **already implement** sleep/wake:

- `VLLMDriver.sleep(service, level)` → `POST /sleep` with level
- `VLLMDriver.wake(service)` → `POST /wake`
- `LlamaCppDriver.sleep()` → `POST /models/unload`
- `LlamaCppDriver.wake()` → `POST /models/load`
- `VLLMDriver.supports_sleep` → `True`
- `LlamaCppDriver.supports_sleep` → `True`

**Gap**: `ServiceManager.switch_mode()` does not call these methods. It only
uses `LifecycleManager.start()` / `.stop()` which map to `systemctl start/stop`.

## Proposed Design: Sleep-Aware Mode Switch

### Current flow (slow)

```
mode switch code → rag:
  1. systemctl stop glm-code     (kill process)     ~1s
  2. systemctl start vllm-embedding (cold boot)     ~30s
  Total: ~31s
```

### Proposed flow (fast)

```
mode switch code → rag:
  1. driver.sleep(glm-code)       (unload model)    ~1s
  2. driver.wake(vllm-embedding)  (if sleeping)      ~instant
     OR systemctl start           (if not running)   ~30s
  Total: ~2s (if services already running in sleep)
```

### Key Insight: Keep All Services Running

Instead of stopping services when leaving a mode, **sleep them**. This means:

1. All sleep-capable services stay running as processes (minimal CPU/RAM)
2. Mode switch only calls sleep/wake HTTP APIs (~instant)
3. Only non-sleep services (embedding, tts) need full stop/start
4. First-time service start still requires systemctl (cold boot)

### Service Lifecycle States

```
STOPPED → (systemctl start) → RUNNING → (sleep) → SLEEPING → (wake) → RUNNING
                                 ↑                                ↓
                                 └──── (systemctl stop) ──────────┘
```

### Implementation Touch Points

1. **`ServiceManager.switch_mode()`** — Add sleep/wake logic before stop/start
2. **`VLLMDriver.sleep()`** — Fix: uses `json={"level": level}` but API expects `?level=N` query param
3. **`VLLMDriver.wake()`** — Fix: uses `/wake` but API endpoint is `/wake_up`
4. **`LifecycleManager`** — Add `sleep()` and `wake()` methods
5. **Health checks** — Sleeping services should report `SLEEPING` state, not `UNHEALTHY`

### VLLMDriver Bug Fixes Needed

```python
# Current (WRONG):
await client.post(f"http://localhost:{service.port}/sleep", json={"level": level})
await client.post(f"http://localhost:{service.port}/wake")

# Correct (per vLLM docs):
await client.post(f"http://localhost:{service.port}/sleep?level=1")
await client.post(f"http://localhost:{service.port}/wake_up")
```

## Estimated Impact

| Scenario | Current | With Sleep | Speedup |
|----------|---------|------------|---------|
| code → rag | ~35s | ~2s | **17x** |
| rag → speak | ~130s | ~35s* | **3.7x** |
| speak → blank | ~5s | ~1s | **5x** |
| code → code (no-op) | ~0s | ~0s | same |

*speak mode has 2 non-sleep services (tts, embedding) that still need cold start

## Risks

1. **CPU RAM usage** — L1/L2 sleeping services keep weights in CPU RAM (~2-8 GB each)
2. **Process count** — More background processes running simultaneously
3. **Stale state** — Sleeping services may lose connection state, need health re-check after wake
4. **Dev mode security** — `/sleep` and `/wake_up` endpoints should not be publicly exposed

## Spike Verification (2026-02-09)

### API Endpoints Confirmed via Context7

| Endpoint | Method | Purpose | Format |
|----------|--------|---------|--------|
| `/sleep?level=1` | POST | L1 sleep (offload to CPU) | Query param, not JSON body |
| `/sleep?level=2` | POST | L2 sleep (discard all) | Query param, not JSON body |
| `/wake_up` | POST | Wake from sleep | Not `/wake` |
| `/wake_up?tags=weights` | POST | Selective wake (L2) | Query param |
| `/is_sleeping` | GET | Check sleep state | Returns `{"is_sleeping": bool}` |
| `/collective_rpc` | POST | RPC for reload_weights | JSON body `{"method":"reload_weights"}` |

### VLLMDriver Bug Confirmation

The existing spike correctly identified the bugs. Context7 docs confirm:

```python
# WRONG (current driver):
await client.post(f"http://localhost:{port}/sleep", json={"level": level})
await client.post(f"http://localhost:{port}/wake")

# CORRECT (verified from Context7):
await client.post(f"http://localhost:{port}/sleep?level={level}")
await client.post(f"http://localhost:{port}/wake_up")
```

### Live Testing Blocked

Live API testing was blocked by:
1. **systemctl --user unavailable** — No D-Bus session in Claude Code environment
2. **VRAM OOM for VL models** — Qwen3-VL-2B needs encoder cache, exhausts 35% util
3. **External process conflicts** — Other vllm instances starting on same ports

### Next Steps

1. **gpumod-b8q**: Fix VLLMDriver endpoints based on confirmed API format
2. **Manual verification**: Test on machine with systemd user session
3. **Integration tests**: Mock-based tests for driver methods

## References

- [vLLM Sleep Mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode)
- vLLM source: `vllm/v1/worker/gpu_worker.py` (sleep implementation)
- gpumod drivers: `src/gpumod/services/drivers/vllm.py`, `llamacpp.py`
- Context7 query: `/websites/vllm_ai_en` (2026-02-09)
