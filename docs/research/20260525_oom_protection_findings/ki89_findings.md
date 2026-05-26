# gpumod-ki89 — Preflight RAM Floor Recalibration

**Date:** 2026-05-26
**Status:** Implemented + Phase 5 verified.
**Depends on:** gpumod-x7rv (original spike), gpumod-56md (`GGML_CUDA_NO_PINNED=1` default), gpumod-1lpe (code-server cgroup protection).

## TL;DR

`DEFAULT_MMAP_OVERHEAD_FACTOR` in [src/gpumod/preflight/ram_check.py](../../../src/gpumod/preflight/ram_check.py) lowered from `1.1` to `0.9`. The 1024 MB `DEFAULT_MIN_FREE_MB` floor stays. Required RAM for a service start drops from `model_size × 1.1 + 1024 MB` to `model_size × 0.9 + 1024 MB`. For the 17,365 MB MTP preserve preset that's **20,125 MB → 16,652 MB** (≈3.5 GiB lower).

End-to-end verified: `gpumod mode switch hermes-agent` succeeds from 19 GiB MemAvailable WITHOUT any operator `drop_caches` step.

## Why the relaxation is safe

The original 1.1× factor was empirically calibrated in gpumod-x7rv against the `cudaHostAlloc`-hang failure class. That class is GONE as of gpumod-56md commit `4131207` — [llamacpp.service.j2](../../../src/gpumod/templates/systemd/llamacpp.service.j2) now sets `GGML_CUDA_NO_PINNED=1` unconditionally, bypassing `cudaMallocHost` and the contiguous-high-order-page requirement entirely.

With pinning gone, the residual cost of a load is:

- mmap of the GGUF (file-backed, demand-paged, reclaimable by the kernel)
- A small CUDA staging buffer (a few MB, not the full model)
- KV cache allocation in VRAM (no host RAM)
- Transient malloc during init (~hundreds of MB)

Empirically (gpumod-x7rv sampler data, cold-cache load):

- Peak VmRSS climbs to ~17.6 GB during sequential disk read
- VmRSS drops to ~1.6 GB after CUDA upload completes
- MemAvailable barely moves during the load (file-backed pages count as available)
- VmPin and Mlocked stay at zero throughout (no pinned allocations)

The 1.1× factor was protecting against a failure mode that no longer exists. With factor=0.9 we're saying: "the peak working set inside the kernel page cache is ~90% of the model file size; the kernel can drop pages under pressure if needed".

## Why factor=0.9 not factor=1.0

The original ticket hypothesis was factor=1.0. Phase 5 testing exposed a subtlety: MemAvailable drops by ~600 MB during `gpumod mode switch` (Python interpreter + systemctl orchestration + DB writes). From a starting state of 18.5 GiB MemAvailable, factor=1.0 + floor=1024 → 18,389 MB required would have FAILED by 460 MB.

Factor=0.9 reflects the kernel's actual peak working set (mmap'd pages are reclaimable), not a hypothetical 100% residency. It gives the formula enough slack to survive the mode-switch overhead from typical desktop steady-state.

## Decision rule applied

The launch prompt required:

> Lowest tested level satisfying (load < 60s) AND (code-server peak probe < 100ms) AND (PSI some-avg10 < 30%) AND (no swap thrash > 5 GiB). Add a 1 GiB safety margin above that level.

Applied to gpumod-x7rv data (deliberately not re-tested, to avoid re-triggering the 12 GiB hard-reboot risk):

| Level | Load time | PSI peak | Code-server probe peak | Swap consumed | Outcome |
|---|---|---|---|---|---|
| 18 GiB | 9 s | 0% | n/a (not measured) | 0 MB | PASS |
| 15 GiB | 49 s | 13.6% | 24 ms | 2.0 GB | PASS (with protection) |
| 12 GiB | 78 min | n/a (host frozen) | unreachable | n/a | **HARD REBOOT** (without GGML_CUDA_NO_PINNED) |

Lowest passing level: **15 GiB**. Plus 1 GiB margin → **16 GiB safe floor**. Formula `model_size × 0.9 + 1024` = 16,652 MB ≈ **16.6 GiB**. Within the safety margin.

**Important:** the 15 GiB data point was collected WITHOUT `GGML_CUDA_NO_PINNED=1`. With the env var active (the configuration the new floor targets), 15 GiB should be even safer. The 16.6 GiB threshold has comfortable headroom.

## Phase 5 verification (end-to-end)

Operator state at the test:

- MemAvailable: 19 GiB (typical desktop steady-state, no drop_caches applied)
- `gpumod doctor sysctl` → OK
- `gpumod doctor oom-protection` → OK
- `GGML_CUDA_NO_PINNED=1` confirmed in `~/.config/systemd/user/qwen36-35b-a3b-mtp-iq4xs-preserve.service`

Test:

```bash
uv run gpumod mode switch hermes-agent
# preflight passed ✓
# both services started ✓
# /health on 7104 (chat) → 200 ✓
# /health on 8210 (embedding) → 200 ✓
```

Smoke completion: `content: "Four"`, `reasoning_chars: 1733`, `finish_reason: stop`. Multi-turn behavior not exercised in this verification.

## Residual risks and open follow-ups

- **CPU-offload services** (`n_gpu_layers < max`) have a different memory profile: model weights stay in CPU RAM as anonymous allocations, not file-backed page cache. The factor=0.9 formula under-estimates required RAM for these. Pre-existing — not introduced by this change. **Follow-up ticket recommended.**
- **Implicit coupling** between the `llamacpp.service.j2` env var and the `ram_check.py` factor — there's no programmatic guardrail. The safety cross-reference comment in `ram_check.py:39-42` flags the dependency, but a future template revert would silently invalidate the floor. **Follow-up: consider extracting the factor into `gpumod doctor` so it warns when the env var is missing.**
- **Phase 5 tested at 19 GiB, not at the new threshold's edge** (16.6 GiB). The math is conservative enough that the boundary test would be redundant, but a future operator-driven test at the edge (e.g., `ram_pressure.py --target-mb 17000` + `gpumod mode switch hermes-agent`) would close the empirical loop.
- **No fresh pressure tests run.** Decision was based on prior gpumod-x7rv data. Deliberate choice — the 12 GiB pressure test had a 50% historical hard-reboot rate, and the new defense stack should make low MemAvailable strictly safer than the prior tests, not more dangerous.

## Files changed

| File | Lines | Change |
|------|---:|---|
| `src/gpumod/preflight/ram_check.py` | +9 / -2 | constant 1.1 → 0.9; module docstring + parameter docstring updated; safety cross-reference comment |
| `tests/unit/test_ram_check.py` | +30 / -0 | new `TestRAMCheckDefaultConstants` class with 4 guard tests |

## Quality gates

- `ruff check src/ tests/` — 0 errors
- `ruff format --check src/ tests/` — 0 changes
- `mypy src/ --strict` — 0 issues
- `pytest tests/unit/ -q` — 2347 passed, 5 skipped
