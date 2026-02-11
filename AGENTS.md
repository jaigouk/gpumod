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

All must pass before any commit:

| Gate | Command | Requirement |
|------|---------|-------------|
| Lint | `ruff check src tests` | 0 errors |
| Types | `uv run mypy src` | 0 errors |
| Tests | `uv run pytest tests/unit -q` | All pass |

### Git Rules

- NEVER commit without explicit user request
- NEVER add Co-Authored-By lines
- NEVER amend unless explicitly asked
- Stage specific files, not `git add -A`

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
git status                    # Check changes
ruff check src tests          # Lint
uv run pytest tests/unit -q   # Tests
bd sync                       # Sync issues
# Only commit/push if user requests
```

## See Also

- [Architecture](docs/ARCHITECTURE.md) — System design (arc42)
- [CLI Reference](docs/cli.md) — Command documentation
- [Presets Guide](docs/presets.md) — Service definitions
- [QA Procedures](docs/qa.md) — Manual testing
