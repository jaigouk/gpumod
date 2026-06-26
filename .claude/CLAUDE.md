# CLAUDE.md

## Project: gpumod

GPU Service Manager for ML workloads. Manages vLLM, llama.cpp, and FastAPI services on NVIDIA GPUs via systemd user units. Includes an MCP server for AI-assistant integration.

## Privacy & Open Source

This is a public open-source repository. **Never include personal information in committed files:**

- No home directory paths (e.g. `/home/<user>/...`) — use `~` or relative paths
- No usernames or real names in issue trackers, configs, or code
- No machine-specific paths in templates, tests, or documentation
- Beads issues (`.beads/issues.jsonl`) must be scrubbed before committing

When writing systemd templates, tests, or documentation, use generic paths like `/opt/gpumod`, `/usr/bin/python3`, or `~/.config/systemd/user/`.

## Architecture

- **Entry point**: `gpumod` CLI (Typer) — `src/gpumod/cli.py`
- **MCP server**: `python -m gpumod.mcp_main` (FastMCP)
- **Systemd templates**: `src/gpumod/templates/systemd/*.j2` (Jinja2)
- **Template engine**: `src/gpumod/templates/engine.py`
- **Service drivers**: `src/gpumod/services/drivers/` (vllm, llamacpp, fastapi, docker)
- **Models**: `src/gpumod/models.py` (Pydantic)
- **DB**: aiosqlite via `src/gpumod/db.py`
- **Presets**: `presets/` — YAML service definitions
- **Modes**: `modes/` — YAML mode definitions
- **Discovery**: `src/gpumod/discovery/` — GPU and model discovery
- **Fetchers**: `src/gpumod/fetchers/` — model fetchers
- **LLM**: `src/gpumod/llm/` — LLM integration
- **Preflight**: `src/gpumod/preflight/` — pre-launch validation (RAM/VRAM/model-file checks; runs from systemd `ExecStartPre`)
- **Doctor**: `src/gpumod/cli_doctor.py` with `src/gpumod/services/{sysctl_check,oom_protection_check,venv_compat}.py` — `gpumod doctor {sysctl|oom-protection|venv}` health checks
- **Host-protection installers**: `scripts/install-gpumod-sysctl.sh` (`vm.min_free_kbytes=1 GiB`, gpumod-ej0) and `scripts/oom-protection/install.sh` (systemd drop-ins for code-server + tuned systemd-oomd, gpumod-1lpe)
- **TUI**: `src/gpumod/tui.py` — terminal UI

## Task Tracking

Use **beads** (`bd`) for all task tracking — epics, features, bugs, spikes, and tasks. Do not use TodoWrite, TaskCreate, or markdown files for project-level tracking.

```bash
bd create --title="..." --type=feature --priority=2   # types: epic, feature, task, bug, spike
bd ready                                                # find available work
bd update <id> --status=in_progress                     # claim work
bd close <id> --reason="..."                            # complete work
```

- Create a beads issue **before** writing code
- Mark `in_progress` when starting, `bd close` when done
- Use dependencies (`bd dep add`) to sequence work
- Scrub PII from `.beads/issues.jsonl` before committing (see Privacy section)

## Testing & Quality Gates

```bash
uv run pytest tests/                    # full suite
uv run pytest tests/unit/               # unit only
uv run ruff check src/ tests/           # lint
uv run ruff format --check src/ tests/  # format check
uv run mypy src/ --strict               # type check
```

| Gate   | Command                                | Requirement |
|--------|----------------------------------------|-------------|
| Lint   | `uv run ruff check src/ tests/`        | 0 errors    |
| Format | `uv run ruff format --check src/ tests/` | No changes |
| Types  | `uv run mypy src/ --strict`            | 0 errors    |
| Tests  | `uv run pytest tests/unit/ -q`         | All pass    |

- All tests use pytest with pytest-asyncio
- **TDD is mandatory**: write the failing test first, watch it fail, then write minimal code to pass
- Red-green-refactor cycle — no production code without a failing test
- Tests must pass and lint must be clean before closing tickets
- Verify with fresh test/lint runs before claiming completion
- **Pre-commit hook** (`scripts/pre-commit-check.sh`) enforces all gates automatically

## Running Long Benchmarks

Benchmarks under `scripts/run_*_benchmark.py` and `docs/benchmarks/**/*.py` typically run for hours. Always run them inside `tmux` and use a **separate tmux session** for live monitoring so an SSH disconnect or terminal crash does not lose a multi-hour run, and so OOM / VRAM pressure can be caught early.

**Launch pattern:**

```bash
# Session 1: the actual benchmark
tmux new -s bench
uv run python scripts/run_qwen36_benchmark.py --model <id> --output-dir docs/benchmarks/<DATE>_<NAME>/ \
    2>&1 | tee docs/benchmarks/<DATE>_<NAME>/run.log
# detach: Ctrl-b d
```

```bash
# Session 2: live monitor (separate session, not just another pane)
tmux new -s monitor
# Split into 3 panes (Ctrl-b ", Ctrl-b %), one each:
nvidia-smi -l 5                                              # VRAM + utilization
journalctl --user -u <service-id>.service -f                 # llama-server / vllm logs
watch -n 5 'free -h && echo --- && dmesg | tail -3'          # RAM + kernel OOM signals
```

**Why two sessions, not panes of one:**
- A panicked `tmux kill-session bench` from the wrong session aborts only the monitor, not the run
- Different attach lifetimes — you can leave the bench session detached for hours while frequently re-attaching the monitor

**When to intervene:**

| Signal | Action |
|--------|--------|
| `nvidia-smi` shows `<500 MiB free` for >30 s | Stop benchmark; you are about to OOM mid-iteration. Identify the co-tenant service, stop it, restart benchmark from clean VRAM |
| `dmesg` shows `Out of memory: Killed process` | Restart from a clean state; the killed process may have been a benchmark dependency (CUDA driver, the service, etc.) |
| Service `/health` returns non-200 for >60 s | Service crashed mid-benchmark; runner will keep firing requests at a dead port. Stop benchmark, restart service, restart benchmark |
| `free -h` shows MemAvailable < 2 GiB | Build a longer cushion before starting; concurrent compiles or other heavy commands will OOM-kill at random |

**Isolation for benchmarks — stop everything that uses the GPU, server-side AND client-side, before launching:**

1. **GPU-resident services** (vLLM, llama.cpp, anything holding VRAM): `gpumod service stop <id>` for each, or `gpumod mode switch blank`. Co-tenant services contaminate TPS via PCIe contention and shrink VRAM headroom for activations.
2. **Long-running agents / orchestrators / IDE assistants** that send requests to GPU-backed endpoints during the run. These are NOT visible from `gpumod service list` (they live outside gpumod's systemd units), but they are GPU consumers: as long as they hold an open client to a live `/v1/chat/completions` port they will keep firing requests at it. Stop the bench-runner script must include shutting them down before the run starts and restarting them after, and the bench script should script that itself rather than relying on operator memory.

Why both, not just (1): even after `mode switch blank`, the bench script then starts the model-under-test on the same port the daily agent was using — so the agent reconnects immediately and starts issuing traffic alongside the benchmark. Symptoms range from contaminated TPS to silent host freezes (no kernel log, no OOM, requires hard reboot) when the client churn coincides with a scheduled background task (e.g. ZFS snapshots, cron) and fragments physical pages out from under the NVIDIA driver. The fix is not to mask the background task — it will be different next time — but to remove the unrelated GPU consumer so the run stands alone.

See feedback memory `feedback_benchmark_vram_isolation.md` for the host-specific consumer-name list and the shutdown sequence.

## Host Stability (cudaHostAlloc-class freezes)

The dominant failure mode on shared GPU hosts is `cudaHostAlloc` hanging the NVIDIA driver when contiguous high-order pages are unavailable. This is **NOT OOM**: no kernel OOM log, no `systemd-oomd` trigger, no PSI signal — the kernel is stuck inside the NVIDIA allocator in uninterruptible I/O wait. See [docs/research/20260525_oom_protection_findings/FINDINGS.md](../docs/research/20260525_oom_protection_findings/FINDINGS.md) for the full incident log.

The defense stack, in priority order:

1. **`GGML_CUDA_NO_PINNED=1` is default for all llamacpp services** (gpumod-56md, 2026-05-26). The Jinja template at `src/gpumod/templates/systemd/llamacpp.service.j2` sets it unconditionally; `cudaMallocHost` is bypassed; the freeze class is eliminated. Measured cost: 0.28% TPS regression. Do NOT remove this line without compelling evidence.
2. **Preflight RAMCheck** (`src/gpumod/preflight/ram_check.py`, gpumod-lgt): refuses service starts when MemAvailable is below `model_size × 1.1 + 1024 MB`. Empirically calibrated; do not relax without a fresh GGML_CUDA_NO_PINNED benchmark on the new floor.
3. **`vm.min_free_kbytes=1 GiB`** (gpumod-ej0): installer at `scripts/install-gpumod-sysctl.sh`. Doctor check: `gpumod doctor sysctl`.
4. **Cgroup memory protection for code-server** (gpumod-1lpe): installer at `scripts/oom-protection/install.sh`. Doctor check: `gpumod doctor oom-protection`. Keeps the operator's lifeline alive during expected pressure events (15 GiB MemAvailable range); does NOT help in the unrecoverable 12 GiB driver-hang case.

**Key counter-intuitive facts to remember:**

- NVIDIA's `cudaHostAlloc` page-locking is **invisible** to `/proc/<pid>/status:VmPin` and `/proc/meminfo:Mlocked`. Don't conclude "no pinned allocation happens" from those counters being zero.
- Swap does NOT satisfy `cudaHostAlloc` directly (page-locked memory is non-swappable), but it CAN absorb anonymous app RSS to free physical RAM for the GGUF page cache. The bd memory `swap-does-not-help-llm-loading-on-this` was corrected 2026-05-25 to clarify this.
- `cudaHostAlloc` failures hang the driver silently; they do NOT raise OOM. Cgroup `memory.high` and `systemd-oomd` cannot catch them.

## Embedding driver

Embeddings on this host run under llama.cpp `--embedding`, not vLLM.
`presets/embedding/gguf-embedding-code.yaml` (preset id
`gguf-embedding-code`, port 8210) serves Qwen3-Embedding-0.6B Q8_0
from `~/bin/Qwen3-Embedding-0.6B-Q8_0.gguf`. The `hermes-agent`, `code`,
and `finetuning` modes reference it; the `rag` and `hacker` modes still
reference the vLLM preset and are intentionally not migrated. Pooling
mode is `--pooling last` (Qwen3-Embedding's training pooling — do not
change without changing the model). Server returns 1024-dim L2-normalised
vectors; callers that need shorter embeddings must slice + L2-renormalise
client-side. See [docs/research/20260606_embedding_llamacpp_vs_vllm/FINDINGS.md](../docs/research/20260606_embedding_llamacpp_vs_vllm/FINDINGS.md)
for the full reasoning (b4t RAM constraint, measured RSS/VRAM/cold-start
deltas, caller audit, what we kept and what we gave up).

## Template-Touching Tickets (mandatory acceptance step)

For any ticket that modifies a file under `src/gpumod/templates/`, the acceptance criteria MUST include:

```bash
uv run gpumod template install-all --yes
```

This re-renders every registered service against the modified template and surfaces preset bugs that the unit tests miss. Lesson learned in gpumod-56md (2026-05-26): the team's PR passed all gates (ruff, format, mypy, 2343 pytest), but `install-all` failed on the first preset that lacked `unit_vars.model_path`. That bug had been latent for months because no team had run `install-all` end-to-end against the full preset matrix.

Same rule applies for any change to `presets/llm/*.yaml` schema expectations: re-render and verify.

## Design Principles

Follow **SOLID**:

- **S**ingle Responsibility: each module/class does one thing (e.g. `mcp_server.py` separates tools, resources, and server lifecycle)
- **O**pen/Closed: extend via new drivers/templates, not by modifying existing ones
- **L**iskov Substitution: drivers are interchangeable behind the same interface
- **I**nterface Segregation: small, focused interfaces (e.g. `TemplateEngine` vs `UnitFileInstaller`)
- **D**ependency Inversion: depend on abstractions — inject `db`, `template_engine`, `unit_installer` into services

## Git Rules

- NEVER commit without explicit user request
- NEVER add Co-Authored-By lines
- NEVER amend unless explicitly asked
- Stage specific files, not `git add -A`

## Conventions

- Python 3.12+, `from __future__ import annotations`
- Use `uv run python` or `uv run pytest` (uv + pyproject.toml)
- Ruff for linting and formatting
- Typer for CLI commands
- Pydantic for models and validation
- Jinja2 sandboxed templates for systemd unit generation
- pytest + pytest-asyncio for testing
- User-level systemd units (`WantedBy=default.target`, no `User=` directive)
- All templates must include `StartLimitBurst` and `StartLimitIntervalSec` in `[Unit]`

## Documentation Updates

Update docs when changing:

- CLI commands → `docs/getting-started/cli.md`
- Service behavior → `docs/architecture/index.md`
- Preset format → `docs/internal/presets.md`
- MCP tools → tool docstrings + `docs/architecture/index.md`

## See Also

- [Architecture](docs/architecture/index.md) — System design (arc42)
- [CLI Reference](docs/getting-started/cli.md) — Command documentation
- [MCP Workflows](docs/user-guide/mcp-workflows.md) — AI assistant usage patterns
