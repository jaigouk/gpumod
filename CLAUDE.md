# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Detailed Project Guidance

The detailed AI-agent guidance lives in [`.claude/CLAUDE.md`](.claude/CLAUDE.md).
Read that first — it contains the full architecture, conventions, testing
matrix, long-benchmark protocol, and stability rules.

## Build & Test

```bash
uv sync                                 # install dependencies
uv run pytest tests/unit/ -q            # unit tests (must pass before close)
uv run ruff check src/ tests/           # lint
uv run ruff format --check src/ tests/  # format check
uv run mypy src/ --strict               # type check
```

The pre-commit hook at [scripts/pre-commit-check.sh](scripts/pre-commit-check.sh)
runs all four gates automatically.

## Architecture Overview

- **CLI** (Typer) → [src/gpumod/cli.py](src/gpumod/cli.py)
- **MCP server** (FastMCP) → `python -m gpumod.mcp_main`
- **Service drivers** → [src/gpumod/services/drivers/](src/gpumod/services/drivers/) (vllm, llamacpp, fastapi, docker)
- **Systemd templates** → [src/gpumod/templates/systemd/*.j2](src/gpumod/templates/systemd/) (sandboxed Jinja2)
- **Presets** → [presets/](presets/) (YAML service definitions)
- **Modes** → [modes/](modes/) (YAML service bundles)
- **DB** → aiosqlite via [src/gpumod/db.py](src/gpumod/db.py)
- **Preflight** → [src/gpumod/preflight/](src/gpumod/preflight/) (RAM/VRAM/model-file checks; runs from `ExecStartPre`)
- **Doctor** → [src/gpumod/cli_doctor.py](src/gpumod/cli_doctor.py) (`sysctl`, `oom-protection`, `venv` subcommands)
- **Host-protection drop-ins** → [scripts/oom-protection/](scripts/oom-protection/), [scripts/install-gpumod-sysctl.sh](scripts/install-gpumod-sysctl.sh)

See [docs/architecture/index.md](docs/architecture/index.md) for the full
system design (arc42 format).

## Conventions & Patterns

- **TDD mandatory** — write the failing test first; no production code
  without a failing test.
- **SOLID** for module design — Single Responsibility, Open/Closed, etc.
- **Beads** (`bd`) for ALL task tracking — never TodoWrite, never markdown
  TODO lists.
- **Privacy** — no home directory paths, usernames, or machine-specific
  paths in committed files. See [.claude/CLAUDE.md](.claude/CLAUDE.md)
  Privacy section.
- **Quality gates** — all four (lint, format, types, tests) must pass
  before any ticket closes.
- **Long benchmarks** — always run inside `tmux` with a separate monitor
  session (see [.claude/CLAUDE.md](.claude/CLAUDE.md) "Running Long
  Benchmarks").
- **Template-touching tickets** — always run `uv run gpumod template
  install-all --yes` as part of acceptance. The test suite covers the
  template engine but not the full preset matrix; running install-all
  surfaces latent preset bugs that pytest misses.

For everything else (commit messages, file ownership rules, MCP tool
patterns, etc.) refer to [`.claude/CLAUDE.md`](.claude/CLAUDE.md).
