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
- **Preflight**: `src/gpumod/preflight/` — pre-launch validation
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

**VRAM isolation for benchmarks:** stop ALL other GPU-resident services before launching (`gpumod service stop <id>` for each). Co-tenant services contaminate TPS measurements via PCIe contention and shrink the headroom for model activations. See feedback memory `feedback_benchmark_vram_isolation.md`.

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
