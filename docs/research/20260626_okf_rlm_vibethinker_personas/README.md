# Research: Local multi-agent reasoning — VibeThinker-3B fit, and an OKF + RLM + persona pivot

**Date:** 2026-06-26
**Status:** Draft — prior experiments documented; open follow-up spike
**Follow-up spike:** gpumod-gos5
**Related:** gpumod-4rbc (hermes-vibe mode)

## Summary

We explored pairing a small local reasoning model (**VibeThinker-3B**) with a generalist
daily driver (**Gemma 4 12B**) on a single 24 GB GPU, to find where the small model adds
value. Two experiments drew a sharp boundary: VibeThinker is strong on **self-contained,
verifiable problems** (best-of-N math/algorithms) but **fails as a multi-file project
developer** — exactly as its model card predicts (it is scoped to competitive-programming /
verifiable reasoning and explicitly **not** to tool-calling / agent / autonomous coding).

That motivates a pivot: instead of using VibeThinker as a *developer*, confine it to
**bounded LEAF reasoning** inside an **OKF + RLM + multi-persona** pipeline that **Gemma
orchestrates**. Whether VibeThinker is genuinely useful as a topical leaf reasoner is the
open question, addressed by the follow-up spike (**gpumod-gos5**).

## Research question (this stage)

Where does a small, reasoning-tuned local model fit alongside a generalist daily driver —
and specifically, is VibeThinker usable as anything beyond a best-of-N math/algorithm solver?

## Setup

`hermes-vibe` mode (gpumod-4rbc): Gemma 4 12B Q5 daily driver on `:7107`, co-resident with a
VibeThinker-3B Q8 **4-slot best-of-N** service on `:7115`. Measured **~20.6 GB used / ~3.5 GB
free** on the 24 GB GPU; a concurrent best-of-4 math query returned the correct answer
(2/3 and 4/4 across two demos) — confirming the co-residency and the best-of-N pattern work.

## Prior experiments

### Experiment 1 — VibeThinker as a multi-file project developer → **FAILED**

A Gemma-led "team" (Gemma = stakeholder + tech-lead/reviewer; VibeThinker = developer + QA)
was tasked to build a small multi-file OKF indexer (index / query / answer + tests) through
**nested loops with REAL gates** — a courier ran `ruff` / `mypy` / `pytest` on the generated
code and fed the actual errors back. Across 2 outer rounds × 2 dev↔QA × 2 dev↔gates
self-checks (~22 min), the gates **never went green**: VibeThinker thrashed on file layout
(top-level vs `src/` vs `tests/`), produced broken relative imports, and invented
non-existent APIs. Even handed the exact error output every iteration, the 3B could not
converge. The orchestration *machinery* worked perfectly; the model was simply out of its
envelope.

### Experiment 2 — minimal OKF indexer written directly → **SUCCEEDED**

A ~120-line stdlib OKF indexer, written directly, built the bundle correctly in one pass
(top-level notes + a multi-part section, each with YAML frontmatter, plus a per-folder
`index.md`). Confirms that "build a multi-file tool" is a generalist/human-shaped task — not
VibeThinker's.

### Cross-check against the model card

The VibeThinker card states it was *"not trained on tool-calling or agent-based programming
data"* and recommends it for *"competitive programming problems (e.g., LeetCode-style)"*; for
hard math (AMOBench) it advises **max tokens 60K–100K**. Our results match exactly:
self-contained verifiable problems succeed; multi-file / agentic coding fails.

## Options considered — VibeThinker's role

| Option | Pros | Cons |
|---|---|---|
| Project developer (multi-file) | local, cheap | ❌ out of envelope; cannot converge even with gate feedback (Exp. 1) |
| Best-of-N verifiable solver (math / algorithms) | ✓ in envelope; cheap, local, fast | narrow; needs a verifier |
| Topical LEAF reasoner over OKF (RLM pattern) | could enrich research/synthesis over a knowledge base | **unknown** — outside the "verifiable" framing; the open spike |

## Recommendation (so far)

- Use **Gemma** (or a human) for multi-file / tool building; do **not** use VibeThinker as a
  project developer.
- Keep **VibeThinker** as a local **best-of-N specialist** for self-contained, verifiable
  problems (competition math, LeetCode-style algorithms).
- The promising direction for a research / knowledge-curation workflow is **OKF + RLM-style
  recursive decomposition with Gemma orchestrating**. Whether VibeThinker adds value at the
  **leaves** (multi-persona topical reasoning) is to be decided by the follow-up spike.

## References

- VibeThinker-3B model card — https://huggingface.co/WeiboAI/VibeThinker-3B
- RLM (Recursive Language Models) — https://github.com/alexzhang13/rlm
- Open Knowledge Format (OKF) — https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing

## Follow-up

- **gpumod-gos5** — *Spike: VibeThinker-3B as a multi-persona LEAF reasoner in an OKF + RLM
  pipeline* (DSPy personas, Gemma synthesis). Self-contained PoC drafted under
  `.notes/inbox-okf-index/` (gitignored): `okf_index.py`, `OKF_SPEC.md`,
  `rlm_persona_spike.py`. The spike's "Spike results" will be appended here on completion.
