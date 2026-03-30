# Beads Spike Template

Use this template for **research/investigation** tasks. A spike is timeboxed exploration
to reduce uncertainty before implementation.

> **Note:** Spikes do **NOT** follow Red/Green/Refactor and do not require production code. Close when research is documented and recommendations (or follow-up tickets) are in place.

```bash
bd create --title="Spike: <Research question>" --type=spike
# Or as a child of an epic:
bd create --title="Spike: <Research question>" --type=spike --parent=<epic-id>
```

---

> **Before Starting:** Always groom the ticket first. Ensure the research question
> is clear, the timebox is realistic, and the expected deliverables are defined.

## Research Question

What specific question(s) need to be answered?

- Question 1
- Question 2

## Timebox

Maximum time allocated for this investigation (e.g., 2 hours, 1 day).

**Timebox:** \_\_\_ hours/days

## Background / Context

- Why is this research needed?
- What decisions are blocked by this uncertainty?
- Links to related issues, docs, or prior discussions.

## Investigation Approach

1. Approach 1 - What will be explored and how.
2. Approach 2 - Alternative approach if first doesn't yield results.

## Research Guidelines for AI Agents

When assigning this spike to an AI agent, include these instructions:

```markdown
Please investigate and provide a clear research report with:

1. **Use Context7** for library/package documentation lookup
2. **Web search** for recent information (use current year in queries)
3. **Structured findings** with pros/cons table for each option
4. **Clear recommendation** with justification
5. **Follow-up tasks** if implementation is needed
```

## Expected Deliverables

- [ ] Research report created at `docs/research/YYYYMMDD_<topic>.md`
- [ ] Recommendation or decision documented
- [ ] PoC code (if applicable, can be throwaway)
- [ ] Follow-up tickets created for implementation (if applicable)

## Findings Template

Copy this template to `docs/research/YYYYMMDD_<topic>.md`:

```markdown
# Research: <Topic>

**Date:** YYYY-MM-DD
**Spike Ticket:** <beads-id>
**Status:** Draft | Final

## Summary

Brief summary of what was discovered.

## Research Question

What question(s) were investigated?

## Options Considered

| Option   | Pros    | Cons    |
| -------- | ------- | ------- |
| Option A | - Pro 1 | - Con 1 |
| Option B | - Pro 1 | - Con 1 |

## Recommendation

What is the recommended path forward and why?

## References

- [Link 1](url) - Description
- [Link 2](url) - Description

## Follow-up Tasks

- [ ] Task 1: Description
- [ ] Task 2: Description
```

## Acceptance Criteria

- [ ] Research question(s) answered
- [ ] Findings documented
- [ ] Recommendation provided (if applicable)
- [ ] Follow-up tickets created (if implementation needed)
- [ ] Timebox respected (or extended with justification)
