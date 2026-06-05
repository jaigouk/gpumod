---
title: Vision-Enabled Workflows
description: Send images to the vision-enabled Gemma 4 preset via OpenAI chat-completions, and know when to use it instead of the multi-slot text-only preset.
---

# Vision-Enabled Workflows

*Learn when to run the vision-enabled Gemma 4 preset instead of the
multi-slot one, and how to send an image through the OpenAI API.*

## Vision vs. multi-slot: pick one

On a 24 GB card you cannot have both. The two Gemma 4 26B-A4B presets trade
the same ~1.1 GiB of VRAM differently:

| | `gemma4-26b-a4b-q4` | `gemma4-26b-a4b-q4-multi` |
|---|---|---|
| Image input | ✅ (`--mmproj` vision encoder loaded) | ❌ text-only by design |
| Slots | 1 × 128K ctx | 3 × 128K ctx |
| Concurrent agents | 1 (others queue) | 3 |
| Use it for | screenshots, diagrams, photo Q&A | agent teams, parallel sessions |

Both serve port `7109`, so switching between them is transparent to your
clients. The multi preset drops the vision encoder *on purpose* — that
1.1 GiB is exactly the headroom the 3×128K KV budget needs (see
[Multi-agent with cont-batching](multi-agent.md)).

## Prerequisites

- gpumod set up — [Run your first service](../getting-started/first-service.md)
- The `gemma4-26b-a4b-q4` unit installed
  (`gpumod template install gemma4-26b-a4b-q4 --yes`)
- Enough free RAM to pass preflight for a ~13 GB model load (see
  [Host stability](host-stability.md))

## Steps

### 1. Swap to the vision-enabled service

If the multi-slot service is running, stop it first (same port, same GPU):

```bash
gpumod service stop gemma4-26b-a4b-q4-multi
gpumod service start gemma4-26b-a4b-q4
```

Expected output:

```
Stopped service gemma4-26b-a4b-q4-multi successfully.
Started service gemma4-26b-a4b-q4 successfully.
```

(If the start is refused with a quiesce-period error, wait the indicated
seconds and retry — the driver is still reclaiming VRAM.)

### 2. Send an image via chat-completions

The OpenAI API carries images as `image_url` content parts. For a local
file, embed it as a base64 data URI:

```bash
IMG_B64=$(base64 -w0 screenshot.png)

curl -s http://localhost:7109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What does this diagram show? Two sentences."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,${IMG_B64}"}}
    ]
  }],
  "max_tokens": 2000
}
EOF
```

Expected output shape (illustrative — content depends on your image):

```json
{
  "choices": [{"message": {"content": "The diagram shows ...",
                           "reasoning_content": "..."},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 350, "completion_tokens": 410}
}
```

Note the high `prompt_tokens`: an image is encoded into hundreds of vision
tokens that occupy KV cache like any other tokens (~200 MiB of VRAM for a
~1000×560 PNG — budget data in the
[capacity research](../research/20260604_multi_agent_hermes_capacity/README.md)).

### 3. Budget tokens for reasoning + vision: `max_tokens >= 2000`

The preset enables thinking (`enable_thinking: true`). For text-only
prompts the floor is `max_tokens >= 1500`
([why](openai-clients.md#the-reasoning-model-gotcha-max_tokens-1500));
**for images, use `>= 2000`** — the model reasons *about the image* before
answering, and describing visual content consumes a noticeably larger
thinking budget. Too small a value exhausts the budget mid-thought and
returns truncated or empty `content` with `finish_reason: "length"`.

### 4. Switch back when you're done

```bash
gpumod service stop gemma4-26b-a4b-q4
gpumod service start gemma4-26b-a4b-q4-multi
```

Or just `gpumod mode switch hermes-agent` — [drift
recovery](modes-deep-dive.md#drift-recovery) re-launches whatever the mode
needs.

## What happened

The `--mmproj` flag loads a separate vision-encoder GGUF alongside the
language model. Incoming `image_url` parts are decoded into vision tokens
by that encoder and inserted into the same KV cache as text tokens — which
is why vision costs both VRAM (encoder weights + image KV) and generation
budget (reasoning about images). gpumod treats all of this as ordinary
preset configuration: the vision preset simply declares a higher `vram_mb`
(19,500 vs 18,000 text-only) and the extra flag.

## Next steps

- [OpenAI-compatible clients](openai-clients.md) — wiring clients that can
  send images (most OpenAI-compatible clients support `image_url`)
- [Multi-agent with cont-batching](multi-agent.md) — the text-only
  trade-off in detail
- [Pick a mode](../getting-started/pick-a-mode.md) — where vision fits in
  the mode catalog
