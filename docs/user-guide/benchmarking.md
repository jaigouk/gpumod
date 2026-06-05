---
title: Benchmark Your Model
description: Run multi-hour model benchmarks safely with tmux and a separate monitor session, and read the results JSONs correctly.
---

# Benchmark Your Model

*Learn how to run a multi-hour benchmark without losing it to an SSH
disconnect or a silent OOM — and how to interpret the numbers it produces.*

## Prerequisites

- A registered service for the model under test —
  [Run your first service](../getting-started/first-service.md)
- `tmux` installed
- The benchmark methodology: [Benchmarks README](../benchmarks/README.md)
  (prompt categories, scoring, hallucination checks)

## Steps

### 1. Isolate VRAM

Stop **all** other GPU services before benchmarking — co-tenant services
contaminate throughput measurements via PCIe contention and shrink the
headroom for the model's activations:

```bash
gpumod mode switch blank
gpumod service start <model-under-test>
```

Expected output:

```
Started service <model-under-test> successfully.
```

### 2. Launch the benchmark inside tmux

Multi-hour runs must survive your terminal. Session 1 holds the run:

```bash
tmux new -s bench
uv run python scripts/benchmark.py \
  --model <service-id> \
  --output docs/benchmarks/<DATE>_<NAME>/ \
  2>&1 | tee docs/benchmarks/<DATE>_<NAME>/run.log
# detach: Ctrl-b d
```

The repository ships several runners under `scripts/`:

| Runner | Use when |
|---|---|
| `scripts/benchmark.py` | Standard quality + performance suite (radar categories, optional consistency checks) |
| `scripts/benchmark_mode_switch.py` | Measuring mode-switch / lifecycle timing |
| `scripts/run_*_benchmark.py` | Model-family-specific long runs (see each script's `--help`) |

### 3. Open a *separate* monitor session

Not a pane in the same session — a second session, so a panicked
`tmux kill-session` aborts only the monitor, never the run:

```bash
tmux new -s monitor
# split into 3 panes (Ctrl-b ", Ctrl-b %), one command each:
nvidia-smi -l 5                                       # VRAM + utilization
journalctl --user -u <service-id>.service -f          # server logs
watch -n 5 'free -h && echo --- && dmesg | tail -3'   # RAM + kernel signals
```

### 4. Know when to intervene

| Signal | Action |
|---|---|
| `nvidia-smi` shows <500 MiB free for >30 s | Stop the benchmark — an OOM mid-iteration is imminent. Find and stop the co-tenant, restart from clean VRAM. |
| `dmesg` shows `Out of memory: Killed process` | Restart from a clean state — the victim may have been a benchmark dependency. |
| Service `/health` non-200 for >60 s | The service crashed; the runner is firing at a dead port. Restart both. |
| `free -h` MemAvailable < 2 GiB | Host is one heavy command away from random OOM kills — build a cushion before continuing. |

A host that freezes *without* any of these signals is the
pinned-memory driver hang — see [Host stability](host-stability.md).

### 5. Read the results

Each run writes JSON files into the output directory (raw responses,
scores, timings) plus a `run.log`. Charts:

```bash
uv run python scripts/benchmark_chart.py \
  docs/benchmarks/<DATE>_<NAME>/*.json \
  --output docs/benchmarks/<DATE>_<NAME>/charts/
```

**The one distinction that prevents most misreadings — aggregate vs
per-call TPS:**

- **Per-call TPS** — tokens/second of a single generation, what one user
  *feels*. Drops as concurrency rises (e.g. 109 → 70 → 43 going from 1 to
  3 to 5 concurrent slots in the
  [multi-agent capacity data](../research/20260604_multi_agent_hermes_capacity/README.md)).
- **Aggregate TPS** — sum across all concurrent generations, the machine's
  total yield. *Rises* with concurrency (109 → 142 → 181 in the same data).

Comparing a single-slot per-call number against a multi-slot aggregate
number is the classic apples-to-oranges error. Reports must also pin the
llama.cpp release tag + commit in their Setup table — TPS is not comparable
across server builds.

## What happened

The two-session tmux pattern separates *run lifetime* from *observation
lifetime*: the bench session stays detached for hours while you re-attach
the monitor freely, and neither an SSH drop nor a wrong `kill-session` can
take down the run. VRAM isolation makes throughput numbers attributable to
the model under test alone, and the intervention table catches the failure
modes that end multi-hour runs silently.

## Next steps

- [Benchmarks README](../benchmarks/README.md) — full methodology, prompt
  categories, scoring
- [Published benchmark reports](../benchmarks/README.md) — see the
  Benchmarks section in the navigation for prior runs and report format
- [Host stability](host-stability.md) — the freeze class that monitoring
  cannot catch
