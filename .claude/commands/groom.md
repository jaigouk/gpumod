---
name: groom
description: Deep-groom a ticket — implementation simulation, scope check, split detection
allowed-tools: Read, Grep, Glob, Bash
---

# /groom <ticket-id>

Deep-groom a single ticket before claiming it. Runs an **implementation simulation** to catch gaps like missing modules, signature mismatches, and scope overload.

## Usage

```
/groom beads-xxx           # Deep-groom one ticket
```

Always groom ONE ticket at a time. Never batch-groom.

## Process

### Phase 1 — Load Context

```bash
bd show <ticket-id>
bd comments <ticket-id>
bd dep list <ticket-id>
bd label list <ticket-id>
```

### Phase 2 — Quick Checks

Verify (can be done from the ticket description alone):

- [ ] **TDD phases** — RED/GREEN/REFACTOR with specific test names and file paths
- [ ] **SOLID mapping** — concrete implementations, not generic placeholders
- [ ] **AC testability** — every acceptance criterion is testable, not vague
- [ ] **No PII** — the ticket body has no real home paths (`/home/<actual-user>/…`),
  usernames, real names, or machine brand/model. bd issues are committed to the
  PUBLIC repo. If any is present, scrub it (`bd update` → `bd export`) before grooming further.

### Phase 3 — Implementation Simulation (THE CRITICAL STEP)

**This is not optional. Read actual code. Trace actual chains.**

#### 3a. Read every referenced source file

For EVERY file the ticket mentions or depends on, use the Read tool:
- Existing modules in the same area
- Models used in signatures (`models.py`)
- Service drivers if applicable (`services/drivers/`)
- Templates if applicable (`templates/`)
- Config and database code if applicable (`config.py`, `db.py`)

**Do NOT say "verified" without citing the file and line you read.**

#### 3b. Trace the dependency chain

For each new class/function the ticket creates, write out:

```
new_function(dep1, dep2)
  → dep1 type: SomeClass at src/gpumod/module.py:NN
  → method called: dep1.method() — confirmed line NN
  → dep2 type: AnotherClass at src/gpumod/other.py:NN
  → imports needed: from gpumod.module import SomeClass
```

If any link is "TBD" or unresolved → **FAIL**.

#### 3c. Verify method signatures

For each method the ticket will call:
- Read the actual definition
- Compare parameter types and return types **exactly**
- Flag mismatches

#### 3d. Cross-reference within the ticket

- Does the design match the actual code structure?
- Do the steps include updating existing tests that will break?
- Does the TDD section have tests for every AC item?

### Phase 4 — Scope Check

Count from the ticket:

| Metric | Threshold | Action |
|--------|-----------|--------|
| New files to create | > 5 | Flag for split |
| New functions/methods | > 20 | Flag for split |
| Modules touched | > 3 | Flag for split |
| Security + business logic | mixed | Must split |

If ANY threshold exceeded → recommend specific split.

### Phase 5 — Report

```
================================================================
GROOMING REPORT: <ticket-id>
================================================================

TDD/SOLID/AC:      [PASS | FAIL]

IMPLEMENTATION SIMULATION:
  Files read:       <N> (list each with path)
  Dependency chain:  [COMPLETE | BROKEN at <link>]
  Signature mismatches: <N>
    - <method>: ticket says X, code has Y (file:line)
  Missing modules:  <N>
  Missing imports:  <N>

SCOPE:
  New files: <N>  |  Functions: <N>  |  Modules: <N>
  [OK | SPLIT RECOMMENDED — <reason>]

================================================================
VERDICT: [READY | NEEDS UPDATE | NEEDS SPLIT]
================================================================
```

## Rules

1. **One ticket at a time.** Never batch-groom multiple tickets in one pass.
2. **Read before you verify.** Every claim must cite the file:line you actually read.
3. **Trace before you approve.** Write out the dependency chain. No shortcuts.
4. **Split before you bloat.** Two focused tickets > one that gets re-scoped mid-implementation.
5. **No false confidence.** If unsure about any step, mark FAIL and say what's unclear.
6. **No PII.** bd issues ship to the PUBLIC repo. Never author or leave real home
   paths, usernames, real names, or machine brand/model in a ticket; scrub on sight.
