# 3-Agent Multi-Slot MTP Validation — Findings (gpumod-knlw)

**Date**: 2026-06-10
**Service**: `gemma4-26b-a4b-qat-mtp-q4-multi` (port 7114, parallel=3 +
cont-batching + mmproj + MTP n-max=4)
**Build**: llama.cpp `fd3271e0b`
**Stimulus**: Sentry `ExternalServices::Errors::DeserializationError`
(PRIVACY-PARTNER-MANAGEMENT-4X CreateTaskJob, transcribed from
operator-supplied screenshot — actual image-input round-trip is out of
scope and tracked separately)

## Test design

Three roles with distinct system prompts: **Detective** (backend
triage), **Architect** (contracts/schemas/design), **SRE** (blast
radius, regression risk, paging). Two rounds:

- Round 1 — each role analyzes the bug independently
- Round 2 — each role sees the other two's Round 1 outputs and either
  (a) escalates a concern others underplayed or (b) explicitly agrees
  and adds the next concrete step

Both rounds fired in 3 concurrent requests against the same llama-server
slot pool. max_tokens=4096 (had to bump from 1024 — thinking-mode block
consumed all tokens at 1024, leaving content empty; finish=length on the
first attempt was the giveaway).

## Performance (sustained load, ~12-21s per round)

Per-slot decode TPS under sustained parallel-3 load:

| Slot | Round 2 decode TPS | Draft acceptance | Generated |
|---|---|---|---|
| 0 | 104.72 | 51.2% (1277/2496) | 1901 toks |
| 1 | 114.92 | 51.2% (923/1804) | 1375 toks |
| 2 | 119.10 | 53.0% (1685/3180) | 2480 toks |

vs the spike's short-prompt smoke (Arm 3): per-slot was ~168 t/s with
77-82% acceptance. **Sustained-load regression: per-slot TPS −33%,
draft acceptance −30 pp.** Acceptance falls because the longer-form
reasoning has more divergent token patterns. Still useful — drafter
saves ~50% of forward passes.

Wall-clock observation: Round 2 ranged 12.7s–21.5s across the 3 slots,
because slot 2's output (Detective) ran 2480 tokens vs slot 1's 1375.
Cont-batching keeps faster slots productive but doesn't parallelize a
single slow request.

## Quality

The transcript (`transcript.md`) is the source of truth. Headline
observations:

1. **Roles stay distinct.** No bleed. Detective stays diagnostic,
   Architect stays structural, SRE stays adversarial.
2. **Cross-pollination is genuine.** Round 2 is not summary —
   SRE Round 2 literally argues against Architect Round 2 ("your
   'clean code' solution is a luxury we can't afford while the
   Sidekiq queue is actively bleeding"). Detective Round 2 explicitly
   escalates the "0 users affected" point that SRE raised in Round 1.
3. **Technical accuracy is good.** All three correctly identify the
   wire-format / contract problem. Architect's DTO recommendation
   (External → Hash → SectionResponse → Section.new) is sound for a
   Ruby/Rails stack.
4. **No hallucinated facts.** Agents do not invent the GitLab issue
   number, the exact deployment timeline, or fake stack frames. When
   they need data they don't have, they ask for it (e.g. Detective:
   "I need the raw JSON response body for a failing request").
5. **Thinking mode is on and helping.** 70-85% of generated tokens are
   in `reasoning_content`, 15-30% in user-visible `content`. The
   reasoning trace is structured and readable. Could be exposed to
   hermes-agent if a deep-debug UI is wanted.

## Failure modes / cautions

- **max_tokens budget**: thinking-mode burns 70-85% of the token budget
  on internal reasoning. At max_tokens=1024 we saw 100% reasoning, 0
  visible content (all finishes were `length`). At 4096 all finishes
  were `stop`. Recommendation: **default max_tokens >= 2048 for
  thinking-mode agents**, or disable thinking per-request via
  `chat_template_kwargs: {enable_thinking: false}` for short-form
  responses (function calls, single-token classifications).
- **No image input verified yet.** The mmproj loaded but only text
  paths were exercised. Issue #23371 (May 2026, llama.cpp) documented
  MTP+Vision OOM during mmproj checkpoint restore at long context —
  verify before relying on image inputs in this preset. Follow-up
  ticket.
- **Sustained-load acceptance is ~50%** vs short-prompt 78%. If MTP is
  marginal in steady state (acceptance × n-max ≤ 1 means net regression
  vs no-MTP), we'd want to measure on a real workload mix. Speculative
  win condition with n-max=4 + 51% acceptance: 2.0 tokens/draft
  attempt — still saving forward passes, but more like a 1.5-1.7×
  speedup, not 1.7-2.2× advertised.

## Suitability for hermes-agent

**At the llama-server layer**: ✅ ready. Aggregate ~340-360 TPS on
sustained 3-concurrent load, VRAM stable at 20 GiB, all three slots
healthy throughout, /health stayed 200 the entire run.

**At the orchestrator layer**: depends on the hermes-agent worker
model. The current `~/.hermes/profiles/orchestrator/config.yaml`
points at port 7110 (single-slot) and uses one base_url for the
gateway. Whether hermes-agent actually FANS OUT N requests to the
backend (using slot concurrency) or SERIALIZES them through a single
HTTP connection is an open question. If it serializes, multi-slot
gives no win — just buys redundancy headroom.

Recommended follow-up before flipping production hermes-agent over
to port 7114:

1. Audit hermes-agent's gateway to see if it actually issues
   concurrent requests for orchestrator subtasks.
2. If yes: change `base_url` to `http://localhost:7114/v1`, restart
   the orchestrator. Confirm `/slots` shows N concurrent slots in use
   during a real agent task.
3. If no: don't ship multi-slot. The current single-slot QAT+MTP
   preset is strictly better for serialized-request loads.

## Decision

**Ship-it / iterate / kill**: **iterate**.

- The multi-slot preset works at the llama-server level.
- Quality on a real production-bug stimulus is high.
- Production swap requires verifying hermes-agent's concurrency
  pattern (separate ticket recommended).
- The n-max=4 finding from the spike also applies here, and per-slot
  acceptance under sustained load is a genuine concern — measure on
  a representative workload mix before committing.

Single-slot `gemma4-26b-a4b-qat-mtp-q4` (port 7110) remains the
production baseline today. The multi-slot preset is now in the repo
as a candidate, ready to test under real hermes-agent traffic once
the orchestrator concurrency question is answered.
