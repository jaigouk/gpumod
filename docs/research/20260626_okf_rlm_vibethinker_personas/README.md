# Research: Local multi-agent reasoning — VibeThinker-3B fit, and an OKF + RLM + persona pivot

**Date:** 2026-06-26
**Status:** Spike **gpumod-gos5 complete** — recommendation **(a): pursue OKF + RLM + VibeThinker-leaves, with a verification layer** (see Spike results)
**Spike:** gpumod-gos5 (results below)
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

## Spike results (gpumod-gos5)

**Run:** 2026-06-26, `hermes-vibe` mode live (Gemma 4 12B Q5 `:7107` + VibeThinker-3B Q8 multi4 `:7115`,
both `/health` ok, ~21.7 GB used / ~2.4 GB free). Topic: `2026-06-21-transformer-attention-research`
(~12 KB bounded OKF context, 6 concept docs). Runner: `.notes/inbox-okf-index/rlm_persona_spike.py`
(DSPy 3.2.1 typed Signatures → VibeThinker leaves; Gemma synthesis). PoC + raw `spike_report.md` are
gitignored under `.notes/inbox-okf-index/`. Well within the 2–4 h timebox (AC6 ✓; run itself ~2–3 min).

### AC1 — pipeline ran end-to-end ✓
OKF bundle → 3 persona leaf calls (VibeThinker) → Gemma synthesis → saved report. The 3 leaves succeeded
on the first pass; the Gemma *combine* step initially failed on a DSPy parse error (see AC3) and succeeded
on a plain-text retry. Report saved.

### Q1 / AC2 — are the VibeThinker leaves useful, non-generic, sound, distinct?
**YES on usefulness and distinctness; mixed on soundness.** VibeThinker is NOT shallow here — it produced
substantive, on-topic, materially different angles, *despite* its card scoping it to verifiable math/code.

| Persona | On-topic | Substantive | Sound | Verdict |
|---|---|---|---|---|
| complexity-analyst | ✓ | ✓ | **partial** | strong but hallucinates specifics |
| skeptic | ✓ | ✓ | ✓ | **strongest** — all 8 critiques defensible |
| implementer | ✓ | ✓ | **partial** | good plan/metrics, wrong core formula |

- **complexity-analyst (sound, with one clear error).** Correct & quantitative: *"the matrix product QK^T
  requires O(n^2 * d) FLOPs … and produces an n by n score matrix that occupies O(n^2) memory (about 4 n^2
  bytes in FP16)"*; *"GQA shares key/value pairs … reducing the total number of stored (k,v) pairs from
  h·n to O(h·b)"*. **Hallucination:** *"Linear attention (Performer, Linformer) … replaces the softmax with
  a solving step (e.g. via L‑BFGS) that introduces a small approximation error on order 10‑3"* — wrong:
  Performer uses random feature maps (FAVOR+), Linformer a low-rank projection; neither uses L-BFGS, and the
  `10⁻³` figure is fabricated.
- **skeptic (fully sound — the best leaf).** *"FlashAttention still computes the same mathematical attention
  but splits it into fast SRAM tiles. It reduces I/O overhead but does not change the theoretical O(n²) work"*
  (correct, and consistent with the analyst); *"compressing information into a fixed-size memory state
  necessarily involves approximation, and the impact on long-range dependency fidelity has not been
  quantified"*. All 8 pushbacks are valid.
- **implementer (good plan, wrong math).** Strong metrics: *"Peak VRAM usage across context lengths (e.g.,
  256 k, 512 k, 1 M, 3 M tokens)"*, *"FLOP count per token (should scale linearly with n)"*. **Error:**
  *"compute scores as `score = W_q * K^T * W_v^T`"* and *"Keep the softmax … Apply the scaled dot-product to
  the linear scores"* — contradicts linear attention, whose point is to avoid the n×n softmax via
  φ(Q)(φ(K)ᵀV) associativity.
- **Distinctness — PASS.** Analyst quantifies and defends; skeptic attacks the *same* claims as unvalidated;
  implementer turns them into a build plan. Independent confirmation: Gemma's synthesis named the
  analyst-vs-skeptic split (*"negligible"* vs *"unquantified"*) as the **sharpest tension** — the angles are
  genuinely complementary, not paraphrases.

### AC3 — DSPy ↔ VibeThinker viability (the surprising result)
**The hypothesized risk inverted.** DSPy's typed Signatures parsed the non-instruction-tuned, `<think>`-heavy
**VibeThinker** cleanly (all 3 leaves `OK`, default `dspy.Predict`/ChatAdapter, no `<think>` leakage in the
`.analysis` field, `max_tokens=12000` sufficient — every output terminated cleanly, no truncation). It was the
instruction-tuned **Gemma** that broke DSPy: ChatAdapter fell back to JSONAdapter, which rejected Gemma's JSON
because it emitted key `"Brief"` while the OutputField is `brief` (case-sensitive) —
*"Expected to find output fields … [brief]; Actual output fields parsed … []"*. The synthesis **content** in
the error payload was high quality; only the strict field-name parse failed.
**Settings/fix that worked:** a plain-text Gemma chat call (no typed field) synthesizes cleanly. Since the
combine output is free-form prose, typed structured output buys nothing there. → Keep DSPy typed Signatures
for the persona **leaves** (typing + future GEPA-style optimization matter there); use plain generation for the
**synthesis** (or a case-tolerant adapter).

### Q4 — is Gemma a competent synthesizer (the RLM "combine")?
**Yes.** It extracted the consensus (vanilla attention fundamentally O(n²)), the sharpest tension
(analyst "negligible" vs skeptic "unquantified"), and a fused next step (implementer's Performer swap +
skeptic's fidelity quantification), each attributed to the persona that raised it. **Caveat:** Gemma
**merged, it did not verify** — it propagated the implementer's garbled Performer formula and never flagged the
analyst's L-BFGS error. Synthesis ≠ fact-check.

### AC4 — Recommendation: **(a) pursue OKF + RLM + VibeThinker-leaves, with a verification layer**
Justified by the above: VibeThinker adds *real, distinct* value at the leaves (esp. the skeptic angle a single
synthesizer would not have generated), so (a) beats (b) "Gemma does the leaves too." Two additions are required
before this is production-worthy, because the leaves invent specifics and the synthesizer does not catch them:

1. **A grounding/verification pass on leaf claims** — a dedicated fact-check persona or retrieval-backed check
   over the OKF context (the leaves hallucinate technical specifics: L-BFGS, the Performer formula, `10⁻³`).
2. **Plain-text synthesis** (drop the typed Signature for combine) or a case-tolerant DSPy adapter.

This recommendation is the operator's call to ratify before follow-up tickets are filed.

## References

- VibeThinker-3B model card — https://huggingface.co/WeiboAI/VibeThinker-3B
- RLM (Recursive Language Models) — https://github.com/alexzhang13/rlm
- Open Knowledge Format (OKF) — https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing

## Follow-up

- **gpumod-gos5** — *Spike: VibeThinker-3B as a multi-persona LEAF reasoner in an OKF + RLM
  pipeline* — **DONE** (see "Spike results" above). Recommendation **(a)**: pursue, with a verification
  layer. PoC under `.notes/inbox-okf-index/` (gitignored): `okf_index.py`, `OKF_SPEC.md`,
  `rlm_persona_spike.py`, `synth_fix.py`.
- **Proposed (pending operator ratification of (a)):**
  - Grounding/verification pass on leaf claims (fact-check persona or retrieval-backed check over the OKF
    context) — the leaves hallucinate specifics (L-BFGS, Performer formula, fabricated `10⁻³`).
  - Plain-text synthesis (or a case-tolerant DSPy adapter) — DSPy's JSONAdapter rejected Gemma's `Brief`
    vs `brief` field casing.
  - Promote the PoC out of `.notes/` into a runnable module if (a) is pursued.
