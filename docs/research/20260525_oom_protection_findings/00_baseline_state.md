# Phase 0 — Baseline Host-Protection State

**Captured:** 2026-05-25 ~03:10
**Ticket:** gpumod-x7rv
**Status:** read-only snapshot; no changes applied yet

## Tooling

| Tool | Available | Version |
|------|-----------|---------|
| `bpftrace` | ✓ | v0.20.2 |
| `strace` | ✓ | (system) |
| `zramctl` | ✓ | (util-linux) |
| `zram-tools` | ✗ | not installed (needs `sudo apt install zram-tools` for Phase 3c) |
| `stress-ng` | ✗ | not installed (Phase 4 uses a python helper instead) |

## systemd-oomd

```
systemd-oomd.service: active (running) since 2026-05-24 23:04:01 CEST
```

`/etc/systemd/oomd.conf` content: only comments + an empty `[OOM]` section. All defaults are kernel/systemd defaults:

| Parameter | Default value | Effect |
|---|---|---|
| `SwapUsedLimit` | 90% | systemd-oomd kills processes in cgroups marked `ManagedOOMSwap=kill` when total swap usage exceeds 90% |
| `DefaultMemoryPressureLimit` | 60% | for cgroups marked `ManagedOOMMemoryPressure=kill`, kill if 10s memory PSI is sustained above 60% |
| `DefaultMemoryPressureDurationSec` | 30s | how long the threshold must be sustained |

`/etc/systemd/oomd.conf.d/` — empty (no drop-ins).

`user@1000.service` properties:

```
ManagedOOMSwap=auto
ManagedOOMMemoryPressure=auto
ManagedOOMMemoryPressureLimit=0
```

`auto` means systemd-oomd *will not* act on this slice unless an ancestor explicitly opts in. **No gpumod service is currently protected by oomd.**

## cgroup v2 layout

Each gpumod service lives under `app.slice`:

```
/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/<service>.service/
```

`user@1000.service/app.slice` memory limits:

| File | Value |
|------|-------|
| `memory.high` | `max` (no throttle) |
| `memory.max` | `max` (no hard cap) |
| `memory.low` | `0` (no protection) |
| `memory.min` | `0` (no guarantee) |

`user@1000.service` memory state at capture:

| Property | Value |
|---|---|
| `MemoryCurrent` | 16.0 GiB |
| `MemoryPeak` | 22.0 GiB |
| `MemorySwapCurrent` | 716 MiB |
| `MemorySwapPeak` | 1.0 GiB |
| `MemoryZSwapCurrent` | 0 (zswap disabled) |
| `MemoryAvailable` | 17.1 GiB |

## Memory subsystem

| Setting | Value | Source |
|---|---|---|
| `vm.swappiness` | 60 | kernel default |
| `vm.min_free_kbytes` | 1,048,576 (1 GiB) | gpumod-ej0 sysctl drop-in |
| `vm.overcommit_memory` | 0 (heuristic) | kernel default |
| ZFS `zfs_arc_max` | 2 GiB | bd memory `swap-does-not-help-llm-loading-on-this` |
| zswap | disabled | `/sys/module/zswap/parameters/enabled` = `N` |
| zram | not loaded | no zram modules |

## Swap configuration

```
Filename       Type        Size       Used      Priority
/dev/zd0       partition   65,536 MiB 3,016 MiB -2     (ZFS zvol)
/dev/sda2      partition   2,048 MiB  0         -3
```

Total swap: 67.6 GiB. Currently 3 GiB used out of that. Priorities differ (zd0 preferred over sda2).

## PSI baseline (no load)

```
some avg10=0.00 avg60=0.00 avg300=0.00 total=13040644
full avg10=0.00 avg60=0.00 avg300=0.00 total=12587455
```

No pressure right now. `total` counters indicate ~13 s of cumulative "some-pressure" time since boot — the rare brief stalls during heavy operations (compiles, model loads). No sustained pressure.

## Memory state at capture

```
               total        used        free      shared  buff/cache   available
Mem:            30 GiB      13 GiB      6.7 GiB   618 MiB 13 GiB       16 GiB
Swap:           65 GiB      2.9 GiB     63 GiB
```

- All gpumod services are stopped (no chat model or embedding running)
- `MemAvailable` is 17.6 GiB — short of the 20.1 GiB preflight requirement by ~2.5 GiB
- This is the steady state of an idle desktop with code-server + a few Claude sessions + browser

## Assumptions captured (to verify)

1. **Preflight formula `model × 1.1 + 1024` is over-cautious** — it was set after a documented OOM hard-reboot, but that incident involved concurrent vllm-embedding holding pinned memory. Loading the MTP model in isolation may be safe with less padding.
2. **Swap doesn't help model loading** (bd memory) — true for the *direct* path (mmap is file-backed, not swap-backed; cudaHostAlloc is pinned). But indirectly swap can absorb idle app RSS, freeing physical RAM for the GGUF page cache.
3. **The host can hard-reboot under OOM** — last verified 2026-05-24 (gpu-stability.md). But that was with the previous protection stack (no app.slice memory.high, no oomd config, no zram).

## Diff target — what Phase 3 will change

| Stage | File | Change |
|---|---|---|
| 3a | `/etc/systemd/oomd.conf.d/gpumod.conf` (NEW) | tighten DefaultMemoryPressureLimit, DefaultMemoryPressureDurationSec |
| 3a | `systemctl set-property user@1000.service/app.slice` | `ManagedOOMMemoryPressure=kill`, `ManagedOOMSwap=kill` |
| 3b | `systemctl set-property user@1000.service/app.slice` | `MemoryHigh=22G` |
| 3c | `sudo apt install zram-tools` + `/etc/default/zramswap` | enable 16 GiB zstd-compressed zram swap |
| 3c | `/etc/sysctl.d/99-gpumod-zram.conf` (NEW) | `vm.swappiness=180`, `vm.page-cluster=0` (zram best practice) |
| 3d | `presets/llm/qwen36-35b-a3b-mtp-iq4xs-preserve.yaml` (TEMP) | add `--no-warmup` to extra_args; will roll back if no benefit |
