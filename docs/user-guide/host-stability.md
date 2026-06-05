---
title: Host Stability and gpumod doctor
description: Understand the cudaHostAlloc freeze class, what GGML_CUDA_NO_PINNED prevents, and how to verify host protection with gpumod doctor.
---

# Host Stability and `gpumod doctor`

*Learn why loading a large model can freeze an entire Linux host without
any OOM message — and how to verify your machine is protected against it.*

## The failure mode: a freeze that isn't OOM

The most dangerous failure on a shared GPU host is **not** running out of
memory. It's `cudaHostAlloc` — the CUDA pinned-memory allocator — hanging
the NVIDIA driver when contiguous physical RAM pages are unavailable.

Symptoms:

- The host becomes unresponsive **during a model load** (SSH stalls,
  editors freeze).
- There is **no OOM log** — `dmesg` is silent, `systemd-oomd` never
  triggers, memory-pressure metrics show nothing. The kernel is stuck in
  uninterruptible I/O wait inside the NVIDIA allocator.
- In the worst case the only way out is a **power cycle**.

Because there is no OOM signal, the usual defenses (cgroup `memory.high`,
`systemd-oomd`, PSI thresholds) **cannot catch it**. Swap doesn't help
either: pinned memory is non-swappable, and swap cannot make contiguous
pages appear in fragmented RAM. The full root-cause investigation is in the
[cudaHostAlloc research findings](../research/20260525_oom_protection_findings/FINDINGS.md)
(gpumod-x7rv).

## The defense stack

gpumod layers four protections, in priority order:

### 1. `GGML_CUDA_NO_PINNED=1` (default for all llama.cpp services)

Every gpumod-rendered llama.cpp unit sets `GGML_CUDA_NO_PINNED=1`
unconditionally. This makes llama.cpp skip `cudaMallocHost` entirely —
no pinned allocation, no contiguous-page requirement, **the freeze class is
eliminated**. The measured cost was a 0.28% throughput regression.

**When not to override:** essentially never. Do not remove this variable
from a unit unless you have re-benchmarked pinned memory on your exact
host and accept the freeze risk.

### 2. Preflight RAM check (runs before every service start)

The systemd unit's `ExecStartPre` runs gpumod's preflight, which refuses to
start a service when `MemAvailable` is below `model_size × 1.1 + 1024 MB`.
This threshold is empirically calibrated: in testing, loads at 18 GiB
available succeeded, while 12 GiB reproduced the hard-reboot freeze. See
[Configuration](../getting-started/configuration.md) for tuning knobs.

### 3. Kernel reserve: `vm.min_free_kbytes`

Keeping a 1 GiB kernel page reserve reduces fragmentation pressure.
Installer: `scripts/install-gpumod-sysctl.sh` (in the repository).

### 4. Editor protection during load spikes

A systemd drop-in gives code-server a guaranteed memory floor so your
editor stays responsive while a model load pushes the host into expected
memory pressure. Installer: `scripts/oom-protection/install.sh`.

## Prerequisites

- gpumod installed — [Getting Started](../getting-started/index.md)
- `sudo` access for the two installers (one-time)

## Steps: verify your host with `gpumod doctor`

### 1. Check the kernel reserve

```bash
gpumod doctor sysctl
```

Expected output:

```
sysctl OK: vm.min_free_kbytes=1048576 kB (>= recommended 1048576 kB)
```

If it fails, install the sysctl drop-in and re-check:

```bash
sudo scripts/install-gpumod-sysctl.sh
gpumod doctor sysctl
```

### 2. Check OOM-protection drop-ins

```bash
gpumod doctor oom-protection
```

Expected output:

```
oom-protection OK: drop-ins installed and directives correct
```

If it fails:

```bash
sudo scripts/oom-protection/install.sh
gpumod doctor oom-protection
```

### 3. Check a service's driver venv

Driver/library version drift (e.g. a vLLM upgrade that breaks a pinned
model) is caught by the venv compatibility check:

```bash
gpumod doctor venv --service-id vllm-embedding-code
```

Expected output: **nothing** — the command is silent and exits `0` when
every package in the service's venv satisfies its `compat:` constraints.
On a mismatch it reports the offending package and exits `1`. See
[Venv Isolation](venv-isolation.md) for how per-service venvs work.

All three checks use exit codes (`0` = OK), so they slot directly into
shell scripts and CI.

## What happened

`gpumod doctor` verifies the *host-level* half of the defense stack — the
parts that live outside gpumod's own process (kernel sysctl, systemd
drop-ins, driver venvs). The *service-level* half (pinned-memory bypass and
RAM preflight) is baked into every rendered unit automatically, so a
correctly installed host plus default presets means the known freeze class
cannot trigger.

!!! warning "One way to reintroduce the freeze"
    Flags that massively inflate VRAM allocation can still wedge the host
    even with all protections installed. The known case: `--swa-full` on a
    Gemma 4 multi-slot preset, which caused a hard-reboot freeze during KV
    allocation. See the warning in
    [Multi-agent with cont-batching](multi-agent.md#when-to-use-the-multi-slot-preset).

## Next steps

- [Configuration](../getting-started/configuration.md) — preflight
  thresholds and environment variables
- [Venv Isolation](venv-isolation.md) — per-service driver venvs
- [Research: cudaHostAlloc freeze findings](../research/20260525_oom_protection_findings/FINDINGS.md)
  — the empirical MemAvailable test matrix behind the thresholds
