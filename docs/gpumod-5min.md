---
marp: true
theme: default
paginate: false
size: 16:9
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

  :root {
    --pop: #e63946;
    --cool: #1d3557;
    --bright: #457b9d;
    --sun: #f4a261;
    --mint: #2a9d8f;
    --bg: #f8f9fa;
    --card: #ffffff;
  }

  section {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: #1a1a2e;
    padding: 56px 80px;
  }

  h1 {
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.05;
    color: var(--cool);
  }

  h2 {
    color: var(--bright);
    font-weight: 600;
  }

  code {
    font-family: 'JetBrains Mono', monospace;
    background: #e9ecef;
    color: var(--cool);
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.88em;
  }

  pre {
    background: var(--cool);
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 8px 32px rgba(29,53,87,0.15);
  }

  pre code {
    background: none;
    padding: 0;
    font-size: 1.15em;
    color: #f1faee;
  }

  em { color: var(--pop); font-style: normal; font-weight: 700; }
  strong { color: var(--mint); font-weight: 700; }

  table {
    font-size: 0.88em;
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }

  th {
    background: var(--cool);
    color: white;
    padding: 14px 20px;
    text-align: left;
    font-weight: 600;
  }

  td {
    padding: 12px 20px;
    border-bottom: 1px solid #e9ecef;
    background: white;
  }

  blockquote {
    border: none;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 16px 28px;
    margin: 20px 0;
    border-radius: 12px;
    font-style: italic;
    color: #495057;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border-left: 5px solid var(--sun);
  }

  ul { list-style: none; padding-left: 0; }

  ul li {
    padding: 10px 0 10px 40px;
    position: relative;
    font-size: 1.05em;
    line-height: 1.4;
  }

  ul li::before {
    content: '!';
    position: absolute;
    left: 8px;
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.75em;
    background: var(--pop);
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    top: 12px;
  }

  /* ===== TITLE ===== */
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, #f1faee 0%, #a8dadc 50%, #e9ecef 100%);
  }

  section.title h1 {
    font-size: 6em;
    color: var(--cool);
    margin-bottom: 0;
  }

  section.title p {
    color: var(--bright);
    font-size: 1.5em;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 12px;
    font-weight: 400;
  }

  /* ===== PROBLEM ===== */
  section.problem {
    background: linear-gradient(160deg, #fff5f5 0%, var(--bg) 50%);
  }

  section.problem h1 {
    color: var(--pop);
    font-size: 2.8em;
  }

  section.problem p:last-of-type {
    color: #868e96;
    font-size: 1.1em;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 2px dashed #dee2e6;
  }

  /* ===== SOLUTION ===== */
  section.solution {
    background: linear-gradient(160deg, #f0fdf4 0%, var(--bg) 50%);
  }

  section.solution h1 {
    font-size: 2.4em;
    margin-bottom: 4px;
  }

  section.solution > p:first-of-type {
    color: #495057;
    font-size: 1.1em;
  }

  section.solution em {
    font-size: 1.3em;
    display: block;
    margin-top: 20px;
    text-align: center;
  }

  /* ===== DEMO ===== */
  section.demo {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 50%, #e8eaf6 100%);
  }

  section.demo h1 {
    color: var(--pop);
    font-size: 5em;
    margin-bottom: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  section.demo h2 {
    font-size: 1.5em;
    font-weight: 400;
    color: #495057;
    font-style: italic;
    margin-top: 8px;
  }

  /* ===== CHEAT ===== */
  section.cheat {
    background: #ffffff;
    padding: 40px 72px;
  }

  section.cheat h1 {
    font-size: 1.8em;
    margin-bottom: 8px;
  }

  section.cheat pre {
    margin: 5px 0;
    padding: 10px 20px;
    background: var(--cool);
    border-radius: 10px;
  }

  section.cheat pre code {
    font-size: 0.85em;
  }

  section.cheat p:first-of-type {
    color: #868e96;
    margin-bottom: 8px;
    font-size: 0.95em;
  }

  /* ===== CTA ===== */
  section.cta {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(135deg, #a8dadc 0%, #f1faee 50%, #a8dadc 100%);
  }

  section.cta h1 {
    font-size: 3.2em;
    color: var(--cool);
    margin-bottom: 20px;
  }

  section.cta p {
    color: var(--bright);
    font-size: 1.2em;
  }

  section.cta pre {
    margin: 24px 0;
    text-align: left;
  }
---

<!-- _class: title -->

# gpumod

Infrastructure for your local AI stack.

<!--
SPEAKER NOTES — SLIDE 1 (10 seconds)

Just the title. Let it breathe.

SAY:
"Hi, I'm [name]. I built gpumod because I got tired of
fighting my GPU every time I wanted to try a new model.
Let me show you what I mean."
-->

---

<!-- _class: problem -->

# The Problem

You find an awesome model on r/LocalLLaMA.
Then you spend the next hour on:

- Will a *quantized version* fit my **24GB**?
- Which quant is *best* for my VRAM budget?
- How do I *swap* it in without breaking what's running?
- Can I run it *alongside* my embedding model?

You shouldn't need a spreadsheet for this.

<!--
SPEAKER NOTES — SLIDE 2 (40 seconds)

SAY:
"Raise your hand if you've ever OOM-crashed trying to load a model.
Yeah. We've all been there.

You see a post on r/LocalLLaMA — someone says this new model is amazing.
But you have 24 gigs. There are 15 different quants.
Q4_K_M? Q5_K_S? Q6_K? Which one actually fits?
And if it fits, can you still run your embedding service next to it?

You end up doing VRAM math in a spreadsheet.
Or worse — you YOLO it and get a CUDA OOM."
-->

---

<!-- _class: solution -->

# Let AI Manage Your GPU

gpumod is an MCP server. You talk to Claude. Claude talks to your GPU.

| You say | gpumod does |
|---------|-------------|
| "Find Qwen3-Coder models" | Searches HuggingFace |
| "What quants fit my machine?" | Filters by your VRAM |
| "Will it fit with embedding?" | Simulates VRAM layout |
| "Switch to code mode" | Stops old, starts new |

*Let me show you.*

<!--
SPEAKER NOTES — SLIDE 3 (40 seconds)

SAY:
"So I built gpumod. It's an MCP server — meaning your AI assistant
can directly search models, check VRAM, and manage GPU services.

Instead of writing systemd units and doing VRAM math,
you just ask in natural language.

Let me show you exactly what this looks like. Live."

TRANSITION: Switch to terminal with Claude Code open.
Make sure gpumod MCP server is connected.
Make sure current mode is "blank" before starting.

PRE-DEMO CHECKLIST:
  gpumod mode switch blank
  # Verify: gpu_status should show ~24GB free, blank mode
-->

---

<!-- _class: demo -->

# LIVE DEMO

## "I found a new model on Reddit. Can I run it?"

<!--
SPEAKER NOTES — SLIDE 4 (~3 minutes)

This is the core. You type prompts into Claude Code.
Claude calls gpumod MCP tools automatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — SEARCH (20 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "Search for Qwen3-Coder GGUF models"

Claude calls: search_hf_models(search="Qwen3-Coder", task="code")
→ Shows list of repos including unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF

SAY:
  "There it is. 30B parameters, Mixture of Experts.
   Let's see which quants fit my 24 gigs."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — FILTER BY VRAM (30 seconds)  ← THE MONEY SHOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "List GGUF files for unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
   that fit my machine"

Claude calls: list_gguf_files(
  repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
  vram_budget_mb=24000
)
→ Filters out quants that won't fit, shows only viable ones

SAY:
  "Out of all those quants, only these fit my card.
   No spreadsheet. No guessing. It just tells me."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — CHECK CURRENT STATE (20 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "What's my GPU status?"

Claude calls: gpu_status()
→ Shows: RTX 4090, 24GB total, blank mode, ~24GB free

SAY:
  "24 gigs free. But I don't just want the coder model —
   I also need my embedding service for code search.
   Will both fit?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — SIMULATE (40 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "Simulate code mode"

Claude calls: simulate_mode(mode_id="code")
→ Shows: qwen3-coder 20GB + vllm-embedding-code 2.5GB = 22.5GB
→ fits: true, headroom: ~2GB

SAY:
  "22.5 out of 24. It fits with 2 gigs to spare.
   No OOM crash. I know before I deploy."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — SHOW ALL MODES (30 seconds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "Show me all available modes"

Claude calls: list_modes()
→ Shows 10 modes: blank, code, rag, nemotron, speak,
   multi-agent-qwen3-coder, multi-agent-qwen3-next, etc.

SAY:
  "I've built up 10 different modes over time.
   Code mode for development. RAG mode for search.
   Multi-agent modes that run 5 parallel sessions.
   Even a speak mode with ASR and TTS.
   One command to switch between any of them."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — SWITCH MODE (30 seconds, optional if time allows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPE:
  "Switch to code mode"

Claude calls: switch_mode(mode_id="code")
→ Starts vllm-embedding-code + qwen3-coder
→ Shows services starting up

SAY:
  "Done. Embedding service is up, coder model is loading.
   One command — gpumod handles the rest."

NOTE: qwen3-coder takes ~30-60s to fully load.
If it's still loading, say:
  "The model is loading now — large models take 30-60 seconds.
   But the point is: I didn't write a single config file."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IF SOMETHING GOES WRONG:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Tool call slow → "Real GPU operations take a few seconds"
- Simulate says won't fit → "See? It just saved me from an OOM crash"
- Service fails to start → "This is why we simulate first"
- Anything weird → TYPE: "Switch to blank mode" (always works as recovery)

TRANSITION: Switch back to slides.
-->

---

<!-- _class: cheat -->

# Demo Cheat Sheet

Copy-paste these into Claude Code, one at a time:

```
Search for Qwen3-Coder GGUF models
```
```
List GGUF files for unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF that fit my machine
```
```
What's my GPU status?
```
```
Simulate code mode
```
```
Show me all available modes
```
```
Switch to code mode
```

Recovery: `Switch to blank mode`

<!--
SPEAKER NOTES — CHEAT SHEET (not shown to audience)

Keep this slide on your second monitor / presenter view.
The audience sees the previous slide (LIVE DEMO).

If you're using Marp presenter mode, this slide is your safety net.
Copy-paste each prompt in order.

TIMING GUIDE:
  1. Search        → 20s  (wait for results, point out the model)
  2. List GGUFs    → 30s  (THE MONEY SHOT — show filtering)
  3. GPU status    → 20s  (show 24GB free, blank mode)
  4. Simulate      → 40s  (show VRAM breakdown, fits: true)
  5. List modes    → 30s  (show 10 modes, narrate variety)
  6. Switch mode   → 30s  (optional — skip if behind on time)

AFTER STEP 6 (or 5 if skipping):
  Switch back to slides → next slide is CTA

IF STUCK:
  "Switch to blank mode" always resets everything.
-->

---

<!-- _class: cta -->

# Take Control of Your GPU

```
$ uv tool install gpumod
$ gpumod init
```

**Open Source. Python. Apache 2.0.**

github.com/jaigouk/gpumod

<!--
SPEAKER NOTES — SLIDE 6 (30 seconds)

SAY:
"That's gpumod.

From Reddit post to running model — search, filter, simulate, deploy.
All through natural language.

It's open source, Apache 2.0, two commands to install.
Link is on the screen.

Questions?"
-->
