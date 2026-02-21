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
- **Service drivers**: `src/gpumod/drivers/` (vllm, llamacpp, fastapi)
- **Models**: `src/gpumod/models.py` (Pydantic)
- **DB**: aiosqlite via `src/gpumod/db/`
- **Presets**: `presets/` — YAML service definitions
- **Modes**: `modes/` — YAML mode definitions

## Task Tracking

Use **beads** (`bd`) for all task tracking — epics, features, bugs, spikes, and tasks. Do not use TodoWrite, TaskCreate, or markdown files for project-level tracking.

```bash
bd create --title="..." --type=feature --priority=2   # types: epic, feature, task, bug, spike
bd ready                                                # find available work
bd update <id> --status=in_progress                     # claim work
bd close <id> --reason="..."                            # complete work
bd sync                                                 # sync with git
```

- Create a beads issue **before** writing code
- Mark `in_progress` when starting, `bd close` when done
- Use dependencies (`bd dep add`) to sequence work
- Scrub PII from `.beads/issues.jsonl` before committing (see Privacy section)

## Testing & Quality Gates

```bash
pytest tests/                    # full suite
pytest tests/unit/               # unit only
ruff check src/ tests/           # lint
ruff format --check src/ tests/  # format check
mypy src/ --strict               # type check
```

| Gate   | Command                        | Requirement |
|--------|--------------------------------|-------------|
| Lint   | `ruff check src/ tests/`       | 0 errors    |
| Format | `ruff format --check src/ tests/` | No changes |
| Types  | `mypy src/ --strict`           | 0 errors    |
| Tests  | `pytest tests/unit/ -q`        | All pass    |

- All tests use pytest with pytest-asyncio
- **TDD is mandatory**: write the failing test first, watch it fail, then write minimal code to pass
- Red-green-refactor cycle — no production code without a failing test
- Tests must pass and lint must be clean before closing tickets
- Verify with fresh test/lint runs before claiming completion
- **Pre-commit hook** (`scripts/pre-commit-check.sh`) enforces all gates automatically

## Design Principles

Follow **SOLID**:

- **S**ingle Responsibility: each module/class does one thing (e.g. `mcp_installer.py` separates detection, rendering, and installation)
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
- Ruff for linting and formatting
- Typer for CLI commands
- Jinja2 sandboxed templates for systemd unit generation
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
