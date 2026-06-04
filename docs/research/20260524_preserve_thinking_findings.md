# preserve_thinking vs enable_thinking — Investigation Findings

**Date:** 2026-05-24
**Ticket:** gpumod-qgy
**Decision:** Hermes-agent (gpumod-aop) will use `preserve_thinking:true` in production. Benchmarked numbers from `result_qwen36-35b-a3b-mtp-iq4xs.json` (`enable_thinking:true`) carry over.

## Question

Does `--chat-template-kwargs '{"preserve_thinking":true}'` change Qwen3.6 MTP behavior vs `--chat-template-kwargs '{"enable_thinking":true}'` on a single-shot prompt? The 76l.3 benchmark used `enable_thinking:true`; we want to know whether the just-measured numbers carry over if production swaps to `preserve_thinking:true` for multi-turn Hermes-agent workflows.

## Method

1. Extract the Qwen3.6 chat template embedded in the MTP GGUF (`tokenizer.chat_template` metadata key).
2. Trace where each kwarg is consumed in the template's Jinja flow.
3. Verify llama.cpp's C++ side does not separately interpret either flag.
4. Cross-check Unsloth + HuggingFace docs for any documented MTP × thinking-flag interaction.

## Findings

### 1. Chat template proof

Extracted via direct GGUF parse from `~/bin/Qwen3.6-35B-A3B-MTP-UD-IQ4_XS.gguf`. The two relevant branches:

**Per-message loop (only fires for PRIOR assistant turns):**

```jinja
{%- elif message.role == "assistant" %}
    {%- set reasoning_content = '' %}
    {%- if message.reasoning_content is string %}
        {%- set reasoning_content = message.reasoning_content %}
    {%- else %}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
    {%- endif %}
    {%- set reasoning_content = reasoning_content|trim %}
    {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
    {%- endif %}
```

The `preserve_thinking is true` test is reachable **only inside the assistant branch**, which only iterates when the conversation has at least one prior assistant message. On a single-shot prompt (`messages = [user]`), the loop never enters this branch.

**Generation prompt block (always emitted, governed only by `enable_thinking`):**

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

`preserve_thinking` is **not referenced** here. Whether the model is told to think on the new turn is decided by `enable_thinking` alone (defaults to "yes" when undefined).

### 2. C++ source confirms no inference-layer special-casing

In `llama.cpp/common/chat.h`:
- `enable_thinking` is a typed field on `common_chat_templates_inputs` (line 190)
- `chat_template_kwargs` is an opaque `std::map<std::string, std::string>` (line 192) — passes through to the Jinja renderer untouched

A repository-wide grep for `preserve_thinking` across the C++ tree returns **zero hits**. The flag exists only inside the template — there is no inference-layer or speculative-decoding-layer code that branches on it.

### 3. Documentation review (no contradicting evidence)

- **Unsloth Qwen3.6 docs**: "preserve_thinking leaves the thinking trace from the previous conversation… increases the number of tokens you use, but could increase accuracy in continued conversations." Qualified language; no measured numbers; no single-shot effect mentioned.
- **HuggingFace model card** (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`): no recommended `chat-template-kwargs` for MTP; no mention of preserve_thinking; no MTP × thinking-flag interaction documented.

## Conclusion

For a single-shot prompt with the Qwen3.6-MTP chat template:

| Aspect | `enable_thinking:true` | `preserve_thinking:true` |
|---|---|---|
| User-message rendering | identical | identical |
| Per-message loop branches taken | none (no prior assistant) | none (no prior assistant) |
| Generation prompt suffix | `<\|im_start\|>assistant\n<think>\n` | `<\|im_start\|>assistant\n<think>\n` |
| MTP draft head input | identical | identical |
| MTP draft head behavior | identical (no code branches on either flag) | identical |
| Expected output distribution | same | same (modulo sampling stochasticity) |

**Single-shot equivalence is proven by construction, not by empirical measurement.**

The two flags diverge only on **continuation turns** where a prior assistant message carries `reasoning_content`:
- `enable_thinking:true` — prior `<think>` is dropped from the templated history
- `preserve_thinking:true` — prior `<think>` is included verbatim in the templated history

This is exactly the multi-turn benefit Unsloth documents ("could increase accuracy in continued conversations").

## Decision for gpumod-aop

Hermes-agent (chat + tool-calling) is multi-turn by nature. The benchmarked numbers from `result_qwen36-35b-a3b-mtp-iq4xs.json` carry over unchanged for production use with `preserve_thinking:true`. The swap therefore lands with the agent-appropriate flag rather than the benchmark-time flag, with zero regression risk on single-shot quality.

Concretely, [presets/llm/qwen36-35b-a3b-mtp-iq4xs.yaml](https://github.com/jaigouk/gpumod/blob/main/presets/llm/qwen36-35b-a3b-mtp-iq4xs.yaml) gets its extra_args updated:

```diff
- extra_args: "--parallel 1 --threads 16 --spec-type draft-mtp --spec-draft-n-max 2 --chat-template-kwargs '{\"enable_thinking\":true}'"
+ extra_args: "--parallel 1 --threads 16 --spec-type draft-mtp --spec-draft-n-max 2 --chat-template-kwargs '{\"preserve_thinking\":true}'"
```

The change is captured under gpumod-aop, not under this spike.

## What was NOT proven

This investigation only proves single-shot equivalence. The actual **value** of `preserve_thinking:true` (accuracy lift on multi-turn agent traces) was not measured — the 20260524 benchmark's single-shot methodology cannot speak to that. If a future ticket wants to quantify multi-turn benefit, it needs a multi-turn evaluation harness, not the v2 coding benchmark.

## Smoke check (runtime confirmation)

The template proof handles correctness. A 1-iteration smoke is the only remaining defensive check — confirms the service starts cleanly with the flag in extra_args and serves a valid response. See gpumod-qgy notes for the smoke runbook.

## References

- [Unsloth Qwen3.6 docs — Thinking, Disable & Preserve Thinking](https://unsloth.ai/docs/models/qwen3.6#thinking-enable-disable--preserve-thinking)
- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF model card](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
- llama.cpp source: `common/chat.h:190-192`, `common/chat.cpp:548-553`
- Chat template extraction: `/tmp/extract_template.py` against the MTP GGUF
- Baseline benchmark: [20260524 Qwen3.6 MTP vs Non-MTP](../benchmarks/20260524_qwen36_mtp_vs_a3b/README.md)
