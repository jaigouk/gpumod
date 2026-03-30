---
name: design-ticket
description: Design a new ticket with architecture verification — architect designs, tech-lead reviews, then creates beads issue
allowed-tools: Agent, Bash, Read, Grep, Glob
---

# /design-ticket <title-or-description>

Design a single ticket with built-in architecture verification. Uses a two-agent chain: architect designs the ticket from codebase analysis, tech-lead reviews it before creation.

## Why This Exists

Creating tickets from rough bullet points produces vague descriptions that fail during implementation. The /groom command catches this AFTER creation — but by then the ticket exists with wrong assumptions baked in. This command prevents bad tickets from being created in the first place.

The key insight: tickets that touch services, drivers, or MCP tools need to trace through existing code (callers, models, templates, DB) BEFORE writing the description.

## Usage

```
/design-ticket Add docker compose driver for multi-container services
/design-ticket MCP tool for GPU memory monitoring
```

Provide a short description of what the ticket should accomplish. The agent will figure out the scope, dependencies, and design from the codebase.

## Process

### Phase 1 — Architect Designs the Ticket

Launch a `developer` agent with this prompt:

> Design a detailed implementation ticket for: {user's description}
>
> Read and analyze:
>
> - `.claude/CLAUDE.md` — project conventions, architecture overview
> - `docs/architecture/index.md` — system design (arc42)
> - Existing code in the relevant area:
>   - `src/gpumod/models.py` — Pydantic models
>   - `src/gpumod/services/drivers/` — service drivers
>   - `src/gpumod/templates/` — Jinja2 systemd templates
>   - `src/gpumod/mcp_tools.py` — MCP tool definitions
>   - `src/gpumod/cli.py`, `src/gpumod/cli_*.py` — CLI commands
>   - `src/gpumod/db.py` — database layer
>   - `src/gpumod/config.py` — configuration
>
> Produce a ticket with these sections:
>
> - **Goal / Problem** — what this ticket accomplishes and why
> - **Background / Context** — with verified file:line references to existing code
> - **Design** — data models, method signatures, sequence flow
> - **SOLID Mapping** — which principles apply and how
> - **TDD Workflow** — RED/GREEN/REFACTOR with specific test names and file paths
> - **Steps** — ordered implementation steps
> - **Acceptance Criteria** — testable criteria
> - **Edge Cases** — boundary conditions to handle
> - **Quality Gates** — ruff, mypy, pytest checks
> - **Risks / Dependencies** — what could go wrong, what this depends on
>
> Key requirements:
>
> - Every file:line claim must be verified by reading the actual file
> - Every method signature must match what's actually in the code
> - Name every dependency (no "TBD" or "to be determined")

### Phase 2 — Tech Lead Reviews the Design

Launch a `tech-lead` agent to review the architect's output:

> Review this ticket design for architecture and SOLID compliance:
>
> {paste architect output}
>
> Check against:
>
> 1. **Architecture alignment**: Does it fit the existing module structure?
> 2. **Dependency direction**: CLI → services → drivers (not the reverse)?
> 3. **SOLID violations**: Any god objects, leaky abstractions, or concrete dependencies?
> 4. **Pydantic models**: Are data boundaries validated properly?
> 5. **Template safety**: Jinja2 sandboxed, no user-controlled templates?
> 6. **Scope creep**: Does it try to do too much? Should it be split?
> 7. **Missing dependencies**: Are there tickets that should exist but don't?
>
> Output: APPROVED, APPROVED WITH FIXES (list fixes), or NEEDS REDESIGN (explain why)

### Phase 3 — Apply Fixes and Create

If tech-lead says APPROVED WITH FIXES:

- Apply the fixes to the ticket description
- Show the user the changes for approval

If tech-lead says NEEDS REDESIGN:

- Show the user the tech-lead's concerns
- Ask whether to redesign or proceed anyway

Once approved:

```bash
bd create --title="<title>" --type=task --priority=2 --description="<ticket body>"
bd dep add <new-id> <depends-on-id>  # for each dependency
```

### Phase 4 — Report

```
================================================================
TICKET DESIGN REPORT
================================================================

TITLE:           <ticket title>
BEADS ID:        <created id>

ARCHITECT:       developer agent
  Files read:    <N>
  Signatures verified: <N>
  Claims cited:  <N> (all with file:line)

TECH LEAD REVIEW:
  Architecture:        [PASS | FIX APPLIED]
  Dependency direction: [PASS | FIX APPLIED]
  SOLID:               [PASS | FIX APPLIED]
  Scope:               [OK | SPLIT RECOMMENDED]

VERDICT: CREATED / NEEDS USER INPUT
================================================================
```

## Rules

1. **One ticket at a time.** Design quality drops when batching.
2. **Architect reads code, tech-lead reviews design.** Don't mix roles.
3. **Every claim is cited.** No "verified" without file:line.
4. **User approves before creation.** Never auto-create tickets.
