# gpumod Agent Instructions

GPU service manager for ML workloads on single-GPU Linux systems.

## Quick Reference

```bash
# Quality gates (must pass before commit)
ruff check src tests && uv run mypy src && uv run pytest tests/unit -q

# Run specific test file
uv run pytest tests/unit/test_manager.py -v

# Format code
ruff format src tests
```

## Project Structure

- `src/gpumod/services/` — Core service layer (manager, lifecycle, drivers)
- `src/gpumod/` — CLI, MCP server, models, database
- `tests/unit/` — Unit tests (1300+, GPU-free)
- `tests/e2e/` — End-to-end tests (require GPU)
- `presets/` — YAML service definitions
- `modes/` — YAML mode definitions

## Development Workflow

### TDD Required

Write tests FIRST, then implementation:
1. **RED** — Write failing test
2. **GREEN** — Minimal code to pass
3. **REFACTOR** — Improve while keeping green

### Quality Gates

All must pass before any commit (enforced by pre-commit hook):

| Gate | Command | Requirement |
|------|---------|-------------|
| Lint | `uv run ruff check src tests` | 0 errors |
| Format | `uv run ruff format --check src tests` | No changes |
| Types | `uv run mypy src --strict` | 0 errors |
| Tests | `uv run pytest tests/unit -q` | All pass |

**Pre-commit hook** (`scripts/pre-commit-check.sh`) runs all checks automatically.
- Commits are **blocked** if any check fails
- Skip tests for faster iteration: `SKIP_TESTS=1 git commit ...`
- Emergency bypass (use sparingly): `SKIP_PRECOMMIT=1 git commit ...`

### Git Rules

- NEVER commit without explicit user request
- NEVER add Co-Authored-By lines
- NEVER amend unless explicitly asked
- Stage specific files, not `git add -A`
- Pre-commit hook ensures quality gates pass

### Web Search Rules

- Current date is 2026 — always use "2026" in search queries
- NEVER use results from 2025 or earlier unless explicitly asked
- Prefer Context7 MCP over WebSearch for library documentation
- When using Context7, pay attention to version numbers in library IDs
- Prefer official documentation over blog posts

## Key Conventions

### Architecture Documentation

[ARCHITECTURE.md](docs/ARCHITECTURE.md) follows arc42 structure. When updating:
- Focus on **what** and **why**, not implementation details
- No version tags (v0.1.1) — that's changelog material
- Use ADR format for decisions (Context → Decision → Consequences)

### Issue Tracking

Use beads for task management:
```bash
bd ready              # Find available work
bd create --title="..." --type=task
bd close <id>         # Mark complete
bd sync               # Push changes
```

## Session Completion

Before ending work:

```bash
# 1. Check what changed
git status

# 2. Verify quality (or let pre-commit hook do it)
./scripts/pre-commit-check.sh

# 3. Sync beads issues
bd sync

# 4. Only commit/push if user explicitly requests
# Pre-commit hook will block if checks fail
```

**The pre-commit hook enforces all quality gates** — you cannot commit
broken code. If a commit is blocked, fix the issue and try again.

## See Also

- [Architecture](docs/ARCHITECTURE.md) — System design (arc42)
- [CLI Reference](docs/cli.md) — Command documentation
- [Presets Guide](docs/presets.md) — Service definitions
- [QA Procedures](docs/qa.md) — Manual testing
