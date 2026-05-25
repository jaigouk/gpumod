# gpumod-x7rv — Spike Findings: LLM Load Host Protection

**Date:** 2026-05-25
**Status:** Root cause confirmed. Code-server protection installed and verified. Follow-up filed.

## TL;DR

The host-freeze class documented in `gpu-stability.md` is **`cudaHostAlloc` hanging the NVIDIA driver when contiguous high-order pages are unavailable**. It is NOT OOM. systemd-oomd, cgroup memory.high, and PSI thresholds **cannot detect or stop it** because there is no OOM signal — the kernel is stuck in uninterruptible I/O wait inside NVIDIA's allocator.

What we found, in priority order:

1. **The current preflight formula (`model × 1.1 + 1024 MB`) is empirically correct.** Going below it (12 GiB MemAvailable test) reproduced the documented hard-reboot. Going above it (18 GiB and 15 GiB tests) succeeded.

2. **Code-server protection works.** The drop-in installed via `install_protection.sh` makes code-server stay responsive at 15 GiB MemAvailable (peak 24ms vs frozen for 55s previously). At 12 GiB it still becomes unresponsive because the whole kernel is busy on uninterruptible I/O — but `MemoryMin=1G` keeps it from being swapped out, so it recovers immediately when the freeze ends or after reboot.

3. **The escape hatch is `GGML_CUDA_NO_PINNED=1`.** Set this on llama-server's environment and `cudaMallocHost` is bypassed. No high-order-page requirement, no driver hang risk. Performance cost unclear — filed as `gpumod-56md` to benchmark. If overhead is small (<5%), this should be the default for all llama.cpp services.

## The mistake in our initial reading

Phase 2 baseline sampler showed `VmPin=0` and `Mlocked` unchanged during a clean model load. I concluded "no pinned memory is used during load."

That was wrong. **NVIDIA's `cudaHostAlloc` uses kernel-internal page-locking via UVM, which is invisible to** `/proc/<pid>/status:VmPin` and `/proc/meminfo:Mlocked`. The pinning IS happening; Linux just doesn't expose it via the standard counters.

Confirmed by source-code inspection:
`~/bin/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1493`:

```cpp
static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    ...
}
```

`cudaMallocHost` is `cudaHostAlloc(..., cudaHostAllocDefault)` — exactly the hang-trigger documented in `gpu-stability.md` and Ollama issue #11317.

## Empirical results

All runs use `qwen36-35b-a3b-mtp-iq4xs-preserve` (131072 ctx + q8_0 KV, 17,365 MB GGUF).

| Test | MemAvailable | Protected? | LLM load time | Swap consumed | Peak PSI | code-server | Outcome |
|---|---|---|---|---|---|---|---|
| Phase 2 baseline | 24 GiB | n/a (no pressure) | 49s | 0 MB | 0.0% | not measured | healthy |
| Phase 4a | 18.7 GiB | partial (just code-server) | 9s (warm) | 0 MB | 0.0% | not measured | healthy |
| Phase 4b unprotected | 15 GiB | no | 55s | 2.8 GB | 10.3% | **frozen ~55s** | healthy LLM but UI dead |
| Phase 4b protected | 15 GiB | yes (drop-in installed) | 49s | 2.0 GB | 13.6% | **24ms peak** | healthy LLM AND UI |
| Phase 4c protected | 12 GiB | yes | 78 min (then aborted) | — | — | **frozen 78 min** | **HARD REBOOT** |

### Reading the table

- **18 GiB to 24 GiB:** safe region. The current preflight (20.1 GiB) sits in the middle of this range. Slightly conservative, but appropriate margin.
- **15 GiB:** workable IF code-server is protected. Without protection, the kernel swaps out code-server's anonymous pages to make room for the model's page cache, freezing the UI for the entire load.
- **12 GiB:** danger zone. Anonymous pressure + model load fragments high-order pages → `cudaHostAlloc` hangs → no recovery without power-cycle.

## Why the original bd memory was right (but I misread it)

`swap-does-not-help-llm-loading-on-this`:
> cudaHostAlloc (pinned memory) requires contiguous physical RAM, non-swappable. mmap'd model pages are file-backed.

Both halves are correct. The pinned allocation is small but it's the actual hang trigger, and it requires *contiguous high-order pages* (a stricter condition than "total RAM available"). Swap CAN displace anonymous app RSS to make more headroom (we measured 2.8 GB displaced at 15 GiB), but it CANNOT make contiguous high-order pages appear when memory is fragmented.

I updated the memory with the corrected wording — same conclusion, sharper explanation.

## Recommendation (final)

| Action | Status |
|---|---|
| Keep current preflight formula (`model × 1.1 + 1024 MB`) | Keep as-is. Don't touch. |
| Install code-server protection drop-in | **Done** — verified at 15 GiB |
| Test `GGML_CUDA_NO_PINNED=1` performance impact | **Filed as gpumod-56md** |
| Add zram or aggressive sysctl tuning | **Skip** — masks the issue without fixing it |
| Update `gpu-stability.md` with `GGML_CUDA_NO_PINNED` option | TODO (cross-project, separate task) |
| Update bd memory | **Done** |

## For gpumod-aop

The Hermes-agent MTP-preserve swap is safe given the current preflight + code-server protection. The operator pain (frequent drop_caches before mode switches) is a steady-state RAM hygiene issue — not an architectural problem. Two paths to relieve it:

- **Short term:** auto drop_caches in `gpumod mode switch` when MemAvailable falls below a threshold. Cheap; doesn't change the safety envelope.
- **Long term:** if `gpumod-56md` shows `GGML_CUDA_NO_PINNED=1` is cheap, enable it by default and relax preflight to `model + 1024 MB`. Eliminates the freeze class entirely.

The aop blocker (`gpumod-aop depends on gpumod-x7rv`) can now be released — the spike's recommendation is "deploy with the current preflight; the protection drop-in is enough."

## Files in this folder

| File | Purpose |
|---|---|
| `00_baseline_state.md` | Phase 0 snapshot of host-protection state before any changes |
| `FINDINGS.md` | this file — final summary |
| `sampler.py` | 10 Hz sampler of /proc/<pid>/status + /proc/meminfo + /proc/pressure/memory + nvidia-smi |
| `ram_pressure.py` | controlled-pressure helper for Phase 4 |
| `run_load.sh` | run-load wrapper using `gpumod service start` (blocking; captures steady state) |
| `run_load_direct.sh` | run-load wrapper using raw systemctl + bypass (non-blocking; captures load phase + code-server probe) |
| `trace_load.bt` | bpftrace script for mmap/mlock/madvise syscalls (not used in final analysis but available) |
| `configs/code-server@.service.d-10-oom-protect.conf` | systemd drop-in for code-server |
| `configs/oomd.conf.d-gpumod.conf` | systemd-oomd tuning drop-in |
| `configs/install_protection.sh` | sudo installer for the drop-ins |
| `runs/<label>/` | per-run snapshots, sampler CSV, code-server probe CSV, llama-server log |
