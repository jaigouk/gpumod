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
   # Poll until model loaded (20B+ models take 30-60s)
   while ! curl -s http://127.0.0.1:7070/models | grep -q '"value":"loaded"'; do
     echo "Waiting for model to load..."; sleep 5
   done
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
   # Poll until model loaded (23GB GGUF takes ~45-60s)
   while ! curl -s http://127.0.0.1:7070/models | grep -q '"value":"loaded"'; do
     echo "Waiting for model to load..."; sleep 5
   done
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

---

## Resilience Testing

This section validates VRAM protection mechanisms that prevent system crashes
from race conditions during mode switches. These tests ensure defense-in-depth
against VRAM exhaustion.

### Background: VRAM Race Condition

CUDA memory release is asynchronous. When `systemctl stop` returns, GPU memory
may still be held by the kernel driver for milliseconds to seconds. Without
proper protection, a subsequent model load can see "30GB needed vs 714MB free"
and crash the system (requiring hard reboot).

### Test 1: VRAM Timeout Failure (Automated)

**Purpose**: Verify mode switch fails gracefully when VRAM not released in time.

```bash
pytest tests/unit/test_manager.py::TestVramTimeoutAborts -v
```

**Expected**:
- All 3 tests pass
- `switch_mode()` returns `ModeResult(success=False)` on timeout
- Error message mentions "VRAM not released"

### Test 2: Sequential Mode Switch Success

**Purpose**: Verify normal operation with proper VRAM wait.

```bash
# Helper function to wait for model load
wait_model() {
  while ! curl -s http://127.0.0.1:7070/models 2>/dev/null | grep -q '"value":"loaded"'; do
    echo "Waiting for model..."; sleep 5
  done
}

# Start fresh
gpumod mode switch blank
nvidia-smi  # Should show ~15MB

# Switch to code mode (20B+ model takes 30-60s)
gpumod mode switch code
wait_model
nvidia-smi  # Should show ~20GB

# Switch back to blank
gpumod mode switch blank
sleep 10  # Wait for VRAM release
nvidia-smi  # Should show ~15MB

# Switch to different GPU-heavy mode (23GB model takes ~45-60s)
gpumod mode switch nemotron
wait_model
nvidia-smi  # Should show ~23GB
```

**Expected**:
- All switches succeed with no errors
- VRAM usage matches expected values
- No warnings about VRAM timeout

### Test 3: Manual curl Protection (CRITICAL)

**Purpose**: Verify system survives direct `/models/load` request with
insufficient VRAM. This tests the driver-level preflight check.

**Prerequisites**:
- Two services configured, each requiring ~20GB VRAM
- Code mode active (using ~20GB)

```bash
# 1. Ensure code mode is active
gpumod mode switch code
nvidia-smi  # Verify ~20GB used

# 2. Start nemotron service WITHOUT gpumod (bypasses manager checks)
systemctl --user start nemotron-3-nano.service
sleep 5  # Wait for server to start (no model loaded yet)

# 3. Attempt manual model load (MUST NOT CRASH SYSTEM)
curl -X POST http://127.0.0.1:7070/models/load \
  -H "Content-Type: application/json" \
  -d '{"model":"Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL"}'

# 4. Verify system stability
nvidia-smi  # Must respond
gpumod status  # Must work
```

**Expected**:
- curl returns error response (JSON or HTTP error), NOT crash
- `nvidia-smi` remains responsive
- System stable, no reboot needed
- Error mentions insufficient VRAM

**If test fails** (system crashes):
- LlamaCppDriver may not have VRAMTracker injected
- Check ServiceRegistry driver initialization

### Test 4: Rapid Mode Switch Stress Test

**Purpose**: Verify stability under rapid sequential switching.

```bash
# Stress test with shorter waits (intentionally aggressive)
# Some failures are acceptable - goal is NO CRASHES
for i in {1..5}; do
  echo "=== Iteration $i ==="
  gpumod mode switch code || echo "code switch failed (acceptable)"
  sleep 20  # Short wait, may not fully load
  gpumod mode switch nemotron || echo "nemotron switch failed (acceptable)"
  sleep 20  # Short wait, may not fully load
done
echo "=== Complete ==="

# Verify system health (MUST SUCCEED)
nvidia-smi
gpumod status
```

**Expected**:
- All iterations complete (success or graceful failure)
- No system crashes or freezes
- Final status shows clean state
- Some switches may fail with VRAM timeout - this is acceptable

### Test 5: Recovery from Failed State

**Purpose**: Verify system can recover from interrupted operations.

```bash
# 1. Simulate stuck state (kill service mid-load)
gpumod mode switch code &
sleep 2
pkill -9 llama-server  # Force kill during load

# 2. Wait for CUDA cleanup
sleep 10

# 3. Attempt recovery
gpumod mode switch blank

# 4. Verify clean state
nvidia-smi  # Should show ~15MB
gpumod status  # Should show blank mode, all services stopped
```

**Expected**:
- Recovery switch succeeds (possibly after VRAM wait)
- System returns to known good state
- No orphan processes
- VRAM fully released

### Test 6: Localhost Binding Verification

**Purpose**: Verify llama-router services not accessible externally.

```bash
# From the server itself (should work)
curl http://127.0.0.1:7070/health

# From another machine or using external IP (should fail)
# Replace <server-ip> with the server's external IP
curl http://<server-ip>:7070/health
```

**Expected**:
- localhost: HTTP 200 OK
- External IP: Connection refused

**Note**: This requires systemd unit files to use `--host 127.0.0.1` instead
of `--host 0.0.0.0`.

### Expected Outcomes Summary

| Test | Pass Condition | Fail Action |
| ---- | -------------- | ----------- |
| VRAM Timeout | Unit tests pass | Check manager.py:208-220 |
| Sequential Switch | All switches succeed | Check VRAMTracker polling |
| Manual curl | Error response, no crash | Verify VRAMTracker injection |
| Rapid Switch | No crashes, graceful errors OK | Check timeout values |
| Recovery | Clean state restored | Check orphan cleanup |
| Localhost Binding | External refused | Update systemd units |

### Resilience Guarantees

After all tests pass, the following guarantees should hold:

1. **ZERO system crashes** from any combination of mode switches
2. **ZERO system crashes** from manual curl to `/models/load`
3. **Graceful failure** with actionable error messages
4. **No data loss** - failed switches leave system in known state
5. **No reboot required** - all failures recoverable via `gpumod mode switch blank`
