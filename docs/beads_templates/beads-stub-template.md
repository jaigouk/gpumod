# Beads Stub Ticket Template

Use this template for **far-term tickets** where blockers haven't been resolved yet.
Full specification will be added when the ticket is promoted (blockers resolved, ticket is near-term).

```bash
bd create --title="Task title" --type=task --parent=<epic-id>
bd dep add <this-ticket> <depends-on>
```

---

> **Stub ticket.** Full specification will be added when blockers are resolved.
> Do not start work on this ticket until it has been promoted to full detail
> using `docs/beads_templates/beads-ticket-template.md`.

## Goal / Problem

<One sentence describing the outcome needed.>

## Module Alignment

| Aspect | Detail |
|--------|--------|
| Module | `src/gpumod/<module>` |
| Layer | CLI / services / drivers / templates / MCP |

## Risks / Dependencies

- Blocked by: <ticket-ids>

> **IMPORTANT:** Dependencies listed here are documentation only. You MUST also set
> formal dependencies with `bd dep add <this-ticket> <depends-on>` so that
> `bd blocked` / `bd ready` / ripple review can see them. Text-only deps are invisible
> to the dependency graph.
