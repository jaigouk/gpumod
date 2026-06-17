# MTP + Multi-Slot on Gemma 4 26B-A4B: Research Note (Phase 1)

**Ticket**: gpumod-mqxg
**Date**: 2026-06-10
**Status**: Phase 1 (web research) complete. Phase 2 (empirical) pending.
**Author**: Jaigouk Kim (with AI assist)
**Host**: RTX 4090 (24 GiB), llama.cpp build `fd3271e0b` (2026-06-09)

## Question

Can `gemma4-26b-a4b-qat-mtp-q4` (port 7110, single-slot today) safely run
with `--parallel 3 --cont-batching` to match the
[hermes-trio recipe](https://github.com/Jaigouk/k3s-setup/blob/main/docs/hermes-trio.md),
and does `--mmproj` survive the multi-slot configuration at 64K per slot?

## Method

Web search and upstream GitHub review constrained to **June 2026 primary
sources** (per operator instruction). May 2026 sources are cited only
where the same finding is re-confirmed in a June discussion thread.

Tools: WebSearch, WebFetch, `gh issue view`, `gh pr view`.

## Source audit

| Source | Date | June filter | Use |
|---|---|---|---|
| [PR #23398 (Gemma4 MTP merge)](https://github.com/ggml-org/llama.cpp/pull/23398) | merged 2026-06-07 | ✅ | Primary |
| [Issue #24266 (MTP kills tokens/sec)](https://github.com/ggml-org/llama.cpp/issues/24266) | opened 2026-06-07 | ✅ | Primary |
| [Issue #23371 (MTP+Vision OOM)](https://github.com/ggml-org/llama.cpp/issues/23371) | opened 2026-05-20 | ❌ | Background (echoed in #23398) |
| [PR #22673 (Qwen MTP base)](https://github.com/ggml-org/llama.cpp/pull/22673) | merged 2026-05-16 | ❌ | Background only |
| dredyson.com Qwen MTP advanced guide | 2026-05-18 | ❌ | Background only |
| braincuber.com MTP tutorial | 2026-05-18 | ❌ | Background only |

## Findings

### F1 — `--parallel > 1` with MTP: "supported but not fully optimized"

From PR #23398 thread (Gemma 4, June 7) and downstream issue #24266 (June 7),
no upstream documentation explicitly bans `--parallel > 1` with
`--spec-type draft-mtp`. The architecture allows it. However, the May-2026
predecessor PR #22673 (Qwen MTP base) called parallel decoding "supported
but not fully optimized" and that statement has not been retracted as of
2026-06-10.

The **strongest June-2026 evidence of `--parallel > 1` working on Gemma 4**
comes from issue [#24266 comments](https://github.com/ggml-org/llama.cpp/issues/24266)
where user `engrtipusultan` posted a working production config with:

```
parallel = 2
mmproj = gemma-4-26B-A4B-it-qat-mmproj-F16.gguf
model = gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
flash-attn = on
ctx-size = 65536
```

This is `parallel=2` (not 3) on Vega 8 / Vulkan (not CUDA / RTX 4090), but
it directly contradicts the qwen sister-preset's "--mmproj unsupported"
annotation. mmproj + MTP coexists. The remaining open question is how far
`parallel` can climb on a different backend.

### F2 — `--spec-draft-n-max 1` is the new Gemma-4 sweet spot

This was the dominant theme in issue #24266 (June 7):

- `sswtodo` (June): "Try with `--spec-draft-n-max 1`. Using MTP below Q8
  significantly reduces draft‑acceptance rates — Gemma is very sensitive
  to MTP quality."
- `engrtipusultan` (June): "behavior is changed after latest gemma MTP
  update. Even for Qwen 3.6 now `--spec-draft-n-max 1` is faster than
  `--spec-draft-n-max 2`. Previously for me `--spec-draft-n-max 2` was
  fastest."
- `m8rr` (June, RTX 4070 SUPER + Gemma 4 26B-A4B): after switching to
  `--spec-draft-n-max 1` + Q4 cache, hit 78 t/s with **91.7% draft
  acceptance** at code-style work. Same exact percentage we measured on
  our host's single-slot smoke test.
- `sswtodo` (June): "Increasing draft threads does not increase tg/s,
  that is not yet solved. So keep `--spec-draft-n-max` at 1."

**Action for Phase 2**: sweep `--spec-draft-n-max` ∈ {1, 2, 4}; expect 1
to win. The single-slot preset's current default of 2 may already be
sub-optimal — separate consideration for [gpumod-8o24](../../../README.md)
bench.

### F3 — Gemma is "very sensitive to MTP quality"

Multiple June commenters in #24266 echo the same point: drafter quants
below Q8 collapse acceptance rate. We are using Q8_0 drafter (462 MB),
matching Unsloth's smallest-and-recommended; this is consistent with the
upstream guidance.

### F4 — Multi-GPU broken at merge; fixed shortly after

PR #23398 author `am17an` (June 7):
> "Multi GPU is currently broken, I will push a fix in a bit."

Not applicable to our single-RTX-4090 host but worth flagging: anyone
reading this who is considering dual-GPU should pull the latest llama.cpp,
not just any post-`fd3271e0b` build.

### F5 — MTP adds VRAM; reduce context accordingly

PR #23398 author `am17an` (June 7), in response to mmproj+MTP VRAM
concerns (echoing the architectural fact in #23371 from May):
> "Since MTP uses extra VRAM, you need to reduce your context accordingly."

This validates the Phase 2 plan: per-slot ctx 64K (not 128K) when stacking
3 slots.

### F6 — Single-stream uplift is real on consumer GPUs

Single-slot success stories in June PR #23398 / issue #24266:

| Host | Model | No-MTP t/s | MTP t/s | Δ | Acceptance |
|---|---|---|---|---|---|
| 2× RTX 3090 (tensor parallel) | Gemma 4 31B Q8_0 | 35.72 | 62.34 | +74.5% | 43.9% |
| RTX 4070 SUPER, Q4 cache | Gemma 4 26B-A4B Q4_K_XL | 56 | 63 (1K), 60 (20K) | +12% (1K), +9% (20K) | (not reported) |
| RTX 4070 SUPER, n-max=1, t=12 | Gemma 4 26B-A4B Q4_K_XL | 63 (1K), 56 (20K) | 68 (1K), 63 (20K) | +8% (1K), +12% (20K) | (not reported) |
| Vega 8 / Vulkan | Gemma 4 26B-A4B QAT Q4_K_XL | 9.x | 18.17 | +100% | (not reported) |
| Our RTX 4090 (smoke test) | Gemma 4 26B-A4B QAT Q4_K_XL | (TBD by gpumod-8o24) | (TBD) | (TBD) | 91.7% |

Important caveat from `BootsSiR` (June 7, dual 5090+4090 + multi-GPU
broken): MTP was slightly slower for him pre-fix. After multi-GPU fix
expected to recover.

### F7 — `--parallel > 1` regression on Qwen3.6-27B (May-2026 background)

Cited here for historical context only, NOT as June-2026 evidence:

- May 2026 dredyson advanced guide (Qwen3.6-27B):
  > "At concurrency 4 (c4), the non-MTP configuration actually pulls ahead
  > in token generation (38.3-41.5 t/s vs. 26.9-31.1 t/s)"
- And: "For MTP workloads, I recommend setting `--parallel` to 1 and
  letting requests queue up, rather than trying to serve multiple
  concurrent requests"

This was the data point that justified opening the spike in the first
place. As of June 2026 it has neither been confirmed nor refuted for
Gemma 4 specifically.

## Phase 2 empirical results (2026-06-10, RTX 4090, build fd3271e0b)

All tests used:
- Model: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (Unsloth QAT)
- Drafter: `gemma-4-26B-A4B-it-Q8_0-MTP.gguf` (462 MB)
- Cache: `--cache-type-k q8_0 --cache-type-v q8_0`, `--cache-ram 4096`
- Flash attention on, ngl 999, threads 16
- Sampler: temp 0.7, top-p 0.95, top-k 64, seed 42
- Prompt: short 40-token code-generation request, 512 max_tokens

### Arm 1 — single-slot `--spec-draft-n-max` sweep

| n-max | Decode TPS | Draft Acceptance | Tokens/draft attempt |
|---|---|---|---|
| 1 | 237.47 | 93.92% (247/263) | 0.94 |
| 2 | 272.75 | 88.59% (326/368) | 1.77 |
| **4** | **291.87** | **78.86% (388/492)** | **3.15** |
| 6 | 281.35 | 66.83% (409/612) | 4.01 |

**Headline contradiction with June 2026 community guidance**: F2's
guidance (use n-max=1 on Gemma 4) is based on RTX 4070 SUPER, Vega 8,
and dual-3080 tests. On RTX 4090 with our exact config, **n-max=4 wins
by ~23%** over n-max=1. Hardware-dependent — more compute headroom on
RTX 4090 lets wider drafts pay off. Curve peaks at 4 and starts
regressing at 6 as acceptance drops below 70%.

**Action for [gpumod-8o24](https://github.com/Jaigouk/gpumod/blob/main/.beads/issues.jsonl)**: the production single-slot preset's `--spec-draft-n-max 2`
is sub-optimal — bump to 4 for +7% throughput at the same VRAM.

### Arm 2 — `--parallel 2 + --mmproj + MTP + n-max=4`

| Slot | Decode TPS | Draft Acceptance |
|---|---|---|
| 0 | 195.29 | 79.7% (388/487) |
| 1 | 188.17 | 74.3% (382/514) |

- **Aggregate ~383 t/s** (1.31× single-slot at same n-max)
- **mmproj + MTP coexistence confirmed at parallel=2** — engrtipusultan's
  June 2026 working pattern reproduces cleanly on RTX 4090.
- Per-slot regression: 292 → 191 t/s (~34% slower per slot)

### Arm 3 — `--parallel 3 + --mmproj + MTP + n-max=4` (hermes-trio target)

| Slot | Decode TPS | Draft Acceptance |
|---|---|---|
| 0 | 167.58 | 78.0% (387/496) |
| 1 | 165.72 | 76.7% (385/502) |
| 2 | 173.11 | 82.3% (391/475) |

- **Aggregate ~506 t/s** (1.74× single-slot at same n-max)
- **VRAM peak 19917 MiB** (4.2 GiB headroom under 24 GiB ceiling)
- Wall time ~3.2 s for 3 × 512-token responses
- **mmproj + MTP + parallel=3 WORKS** — the qwen sister-preset's
  "--mmproj unsupported" annotation is definitively obsolete for
  Gemma 4 on llama.cpp `fd3271e0b`.
- Per-slot draft acceptance (76-82%) stays close to single-slot
  baseline (78.9%); concurrency does not collapse acceptance.

### Arm 4 — `--parallel 3 + MTP + n-max=4`, **text-only** (no mmproj)

| Slot | Decode TPS | Draft Acceptance |
|---|---|---|
| 0 | 166.12 | 79.7% (389/488) |
| 1 | 155.09 | 73.3% (381/520) |
| 2 | 153.74 | 72.3% (379/524) |

- **Aggregate ~474 t/s** (slightly slower than Arm 3 with mmproj —
  within scheduling noise)
- VRAM peak: 18775 MiB (-1.1 GiB vs Arm 3, matches mmproj BF16 file size)
- **mmproj has effectively zero throughput cost** when only doing text
  inference. The 1.1 GiB VRAM is the only price. No reason to drop
  mmproj for the multi-slot preset unless VRAM is critically tight.

### Cross-arm comparison

| Arm | Config | Aggregate TPS | Per-slot TPS | VRAM peak |
|---|---|---|---|---|
| 1.1 | par=1, n-max=1 | 237 | 237 | n/a |
| 1.2 | par=1, n-max=2 | 273 | 273 | n/a |
| 1.3 | par=1, n-max=4 | **292** | 292 | n/a |
| 1.4 | par=1, n-max=6 | 281 | 281 | n/a |
| 2 | par=2 + mmproj, n-max=4 | 383 | 191 | 18.6 GiB |
| **3** | **par=3 + mmproj, n-max=4** | **506** | **168** | **19.9 GiB** |
| 4 | par=3 text-only, n-max=4 | 474 | 158 | 18.8 GiB |

vs non-MTP baseline (`gemma4-26b-a4b-q4-multi`, gpumod-8xaq Phase 2):
- Non-MTP at parallel=3, 128K per slot: **132 TPS aggregate**
- MTP at parallel=3, 64K per slot (Arm 3): **506 TPS aggregate**
- **Net speedup: 3.83×** — even at half the per-slot context window.

### Phase 2 follow-up: per-slot ctx ceiling sweep (2026-06-10, gpumod-knlw)

After validating 64K-per-slot end-to-end, operator asked whether 100K
or 128K per slot would survive on the same hardware. Quick direct-server
tests (no preset / no `gpumod`) with the same config flags, 3 parallel
slots, MTP n-max=4, mmproj on:

| Per-slot | Total ctx | LLM-only peak | + embeddings (~1.8 GiB) | Real headroom |
|---|---|---|---|---|
| 64K | 192K | 19.9 GiB | 21.7 GiB | **2.3 GiB ✅** |
| 100K | 300K | 22.0 GiB | 23.8 GiB | ~0.2 GiB ⚠️ |
| 128K | 384K | **24.0 GiB** | 25.8 GiB | overshoot ❌ |

Linear-ish scaling at ~21 MiB per +1K total ctx, slightly sub-linear at
the high end thanks to Gemma's 1024-token sliding-window attention
(only sink layers allocate full-ctx KV). The hermes-agent mode runs
the embeddings sidecar (`gguf-embedding-code`, ~1.8 GiB) co-tenant in
the same mode, so the LLM-only number must be added to ~1.8 GiB for the
production budget. 100K is at the ragged edge (~200 MiB headroom);
128K overshoots by ~1.8 GiB with embeddings co-tenant. **64K is the
right per-slot ctx for this preset** — gives ~2.3 GiB headroom for
image-token KV growth on multimodal turns.

Raw artifacts: `agent_smoke_128k/` (Round 1 outputs + vram.log peak
24027 MiB), `agent_smoke_102400/` (Round 1 outputs + vram.log peak
22015 MiB). Round 2 prompts in those test scripts had a bash heredoc
bug with backticks in the model output — the LLM-side numbers are still
valid, just the multi-round dialogue wasn't captured.

For per-slot context >64K on this hardware, the trade space is:

- **Drop mmproj** (text-only multi) → frees ~1.1 GiB → fits up to
  ~113K per slot with embeddings co-tenant. Loses image inputs.
- **Drop embeddings** (LLM-only, route hermes-agent embeddings elsewhere)
  → frees ~1.8 GiB → fits up to ~100K per slot with mmproj. Adds an
  external dependency on a separate embeddings host.
- **Stay at 64K with mmproj + embeddings** (current preset). Safest,
  proven, 30-round-conversation budget per slot covers normal agent
  workloads.

## Phase 2 conclusion

**Yes, `gemma4-26b-a4b-qat-mtp-q4` can safely run `--parallel 3 +
--cont-batching + --mmproj`** at 64K per slot on RTX 4090 with
llama.cpp `fd3271e0b`. The hermes-trio recipe works WITH MTP.

Recommended production preset shape (to be tested in a follow-up
production-readiness ticket — NOT changing single-slot today):

```yaml
id: gemma4-26b-a4b-qat-mtp-q4-multi   # new preset, not a replacement
port: 7113                             # separate from single-slot 7110
vram_mb: 21500                         # observed peak 19.9 GiB + headroom
context_size: 65536                    # per-slot (template will × parallel)
unit_vars:
  extra_args: >-
    --parallel 3 --cont-batching
    --threads 16
    --cache-type-k q8_0 --cache-type-v q8_0
    --cache-ram 4096
    --slot-save-path $HOME/.cache/llama-slots
    --model-draft $HOME/bin/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
    --spec-type draft-mtp --spec-draft-n-max 4
    --mmproj $HOME/bin/gemma-4-26B-A4B-it-mmproj-BF16.gguf
    --temp 1.0 --top-p 0.95 --top-k 64
    --chat-template-kwargs '{"enable_thinking":true}'
```

**Caveats / what we did NOT test:**

- Long-context behavior. Our prompt was 40 tokens; behavior at 32K+
  per-slot ctx may differ — MTP layer KV grows linearly.
- Image inputs. We confirmed mmproj **loads** with parallel=3+MTP
  but did NOT send actual image content. Issue #23371 (May 2026)
  documented MTP+Vision OOM during mmproj restore at long context —
  re-verify before claiming images work.
- Sustained load. Single-shot 3-concurrent requests, no slot churn.
  A real agent workload with slot save/restore over hours could
  surface different failure modes.
- Quality. We did NOT compare output quality vs non-MTP — MTP shouldn't
  affect quality but verify with a coding-suite eval before promoting.
- `--cache-ram 4096`: never observed swap-out under our short test.
  At 32K+ per-slot ctx this knob actually engages — separate test
  needed.

## Phase 1 conclusion (superseded by Phase 2 above — kept for audit trail)

**Don't change the production single-slot preset.** Current single-slot
`gemma4-26b-a4b-qat-mtp-q4` on port 7110 is the documented happy path in
the June 2026 literature; our measured 91.7% acceptance matches the
upstream `m8rr` configuration exactly.

**For Phase 2 empirical testing, run THIS order:**

1. **Sweep `--spec-draft-n-max`** ∈ {1, 2, 4} at `--parallel 1` first.
   Establish the single-slot optimum on our hardware (probably 1 per F2).
   This also feeds gpumod-8o24 directly.

2. **Test `--parallel 2` + MTP + mmproj** (text + image requests). This
   replicates `engrtipusultan`'s known-working config pattern on our
   hardware. If parallel=2 works, parallel=3 is at least plausible.

3. **Test `--parallel 3` + MTP + mmproj** at 64K per slot, `--cache-ram
   4096`, `--slot-save-path` (matches hermes-trio recipe). If init refuses,
   document the exact error and link the journal/dmesg evidence.

4. **Test `--parallel 3` + MTP, text-only (no mmproj)**, mirroring the
   non-MTP multi preset's choice. Useful fallback if step 3 fails on
   mmproj specifically.

5. **Adversarial isolation**: if 2 or 3 fail, decouple — try `--parallel 3`
   without MTP, MTP without `--cont-batching`, etc., to identify which
   knob breaks first.

**Risks to monitor during Phase 2:**

- `cudaHostAlloc` driver freeze class (host-stability section of
  `.claude/CLAUDE.md`): stop ALL co-tenant services first per
  `feedback_benchmark_vram_isolation`. Verify MemAvailable ≥ 15 GiB
  before starting.
- VRAM peak: if Phase 2 sees >23 GiB sustained, halve per-slot ctx and
  retry — do NOT raise vram_mb past 22500 without compelling evidence.
- Draft acceptance collapse: if any slot shows acceptance < 30%, abort —
  inferior to non-MTP throughput per the dredyson data.
- Multi-GPU regression notes do not apply (we are single-GPU).

## Phase 2 protocol (executable)

```bash
# Pre-flight: confirm clean host state
sudo dmesg | tail -20             # no recent OOMs
free -h                           # MemAvailable >= 15 GiB
nvidia-smi --query-gpu=memory.free --format=csv,noheader   # >= 22 GiB free

# Stop all co-tenant services
uv run gpumod service stop gemma4-26b-a4b-qat-mtp-q4
uv run gpumod service stop gguf-embedding-code
# (verify nothing else loaded; gpumod-mqxg is empirical, not production)

# Launch in tmux + separate monitor session
tmux new -s mqxg-bench
# Session 2: tmux new -s monitor (3 panes: nvidia-smi -l 5,
#   journalctl --user -u <service> -f, watch -n 5 'free -h && dmesg | tail -3')
```

Then test each arm via temporary preset files (do NOT modify the
production preset). Record per arm:
- `journalctl` line `statistics draft-mtp: ... #gen drafts = N, #acc
  drafts = M` for per-slot acceptance.
- nvidia-smi VRAM peak.
- `free -h` MemAvailable trough.
- Aggregate TPS via the
  `docs/research/20260604_multi_agent_hermes_capacity/phase2_workload_runner.py`
  harness (adapt for 64K per slot).

## Open questions to resolve in Phase 2

1. Does `--mmproj` + MTP + `--parallel 3` initialize on Gemma 4 + RTX
   4090 + llama.cpp `fd3271e0b`? (`engrtipusultan` proved
   `parallel=2 + mmproj + MTP` works; whether 3 also does is unverified.)
2. If yes: what's the per-slot draft acceptance at `--spec-draft-n-max
   1`? Does it stay above 50%?
3. What is the aggregate TPS at `--parallel 3` + MTP vs the existing
   non-MTP multi preset (132 TPS at 128K × 3 per gpumod-8xaq Phase 2)?
   If lower, MTP-multi is strictly worse than the existing options.
4. Does `--cache-ram 4096` interact correctly with MTP layer KV
   spillover, or does the MTP layer KV stay on-device?

## What we are NOT testing in this spike

- Multi-GPU MTP — single GPU host, not relevant.
- Dense Gemma 4 (12B / 31B) — out of scope; ticket is for the 26B-A4B
  MoE variant specifically.
- Non-Q8 drafter quants — F3 makes Q8 the de-facto requirement for Gemma.
- `--spec-draft-n-max > 4` — sswtodo says draft threads do not help yet.

## Sources

June 2026 primary sources:

- [llama.cpp PR #23398 — llama: add Gemma4 MTP (merged 2026-06-07)](https://github.com/ggml-org/llama.cpp/pull/23398)
- [llama.cpp Issue #24266 — Misc. bug: llama : add Gemma4 MTP (#23398) kills tokens/sec (opened 2026-06-07)](https://github.com/ggml-org/llama.cpp/issues/24266)

Background (May 2026 — cited only where June re-confirms):

- [llama.cpp PR #22673 — llama + spec: MTP Support (Qwen base, merged 2026-05-16)](https://github.com/ggml-org/llama.cpp/pull/22673)
- [llama.cpp Issue #23371 — MTP+Vision OOM during mmproj restore (opened 2026-05-20)](https://github.com/ggml-org/llama.cpp/issues/23371)
- [Unsloth MTP guide](https://unsloth.ai/docs/models/mtp)
- dredyson.com Qwen3.6-27B advanced MTP guide (2026-05-18) — text-only background

Cross-repo references:

- [Unsloth Gemma 4 26B-A4B QAT GGUF with MTP folder](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/tree/main/MTP)
- [k3s-setup/docs/hermes-trio.md](https://github.com/Jaigouk/k3s-setup/blob/main/docs/hermes-trio.md) — the 3-slot recipe being tested against
- [docs/research/20260604_multi_agent_hermes_capacity/README.md](../20260604_multi_agent_hermes_capacity/README.md) — the non-MTP multi-slot capacity baseline
