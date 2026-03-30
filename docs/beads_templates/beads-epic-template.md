# Beads Epic Template

Use this template when creating an epic with beads:

```bash
bd create --title="Epic: <Title>" --type=epic --priority=0
```

---

> **Before Starting:** Always groom the epic first. Ensure the goal is clear,
> success metrics are measurable, scope is well-defined, and child tasks are planned.

## Goal / Problem

High-level problem statement and desired outcome.

## Architecture Alignment

Which module(s) does this epic affect? Reference `docs/architecture/index.md`:

| Module | Impact |
|--------|--------|
| `src/gpumod/<module>` | What changes |

## Success Metrics

- Metric 1 (measurable)
- Metric 2

## Scope

**In scope**

- Item 1
- Item 2

**Out of scope**

- Item 1

## Phases / Milestones

1. Phase 1 - Research / design
2. Phase 2 - Implementation
3. Phase 3 - Validation / rollout

## Child Tasks (planned)

Create child tasks under this epic:

```bash
bd create --title="Task title" --type=task --parent=<epic-id>
```

## Dependencies

- Dependency 1
- Dependency 2

## Risks / Unknowns

- Risk 1
- Unknown 1

## Acceptance Criteria

- [ ] All child tasks closed (`bd close <id>`) — each child must have passed quality gates and QA before close
- [ ] Documentation updated where required
- [ ] For all code changes (in child tasks), quality gates were run before each task was closed:
  - [ ] `uv run ruff check src/ tests/` (lint)
  - [ ] `uv run mypy src/ --strict` (type check)
  - [ ] `uv run pytest tests/ --cov=src/gpumod --cov-fail-under=80` (tests with 80% coverage)
- [ ] QA was done for each child task before close (happy path, edge cases, error handling, no regressions)
