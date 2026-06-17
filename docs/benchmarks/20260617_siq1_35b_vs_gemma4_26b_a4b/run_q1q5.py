#!/usr/bin/env python3
"""Send 5 quality items to a llama-server and dump responses."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def call(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--items", default="/tmp/quality_items.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking=False (SIQ-specific).",
    )
    args = ap.parse_args()

    items = json.load(open(args.items))
    out = {"label": args.label, "url": args.url, "params": vars(args), "results": []}
    for it in items:
        msg = [
            {
                "role": "user",
                "content": (
                    it["question"]
                    + "\n\nGive a concise final answer. Show key reasoning steps."
                ),
            }
        ]
        payload: dict = {
            "model": "x",
            "messages": msg,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
        }
        if args.top_k > 0:
            payload["top_k"] = args.top_k
        if args.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        try:
            resp = call(args.url, payload, timeout=900)
            dt = time.time() - t0
            content = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            timings = resp.get("timings", {})
        except Exception as e:
            out["results"].append(
                {"id": it["id"], "error": str(e), "elapsed_s": time.time() - t0}
            )
            print(f"  {it['id']}: ERROR {e}", file=sys.stderr)
            continue
        result = {
            "id": it["id"],
            "domain": it["domain"],
            "expected_key": it["expected_key"],
            "content": content,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "elapsed_s": round(dt, 2),
            "tps_eval": timings.get("predicted_per_second"),
        }
        out["results"].append(result)
        print(
            f"  {it['id']}: {usage.get('completion_tokens')}tok in {dt:.1f}s "
            f"(tps={timings.get('predicted_per_second', 0):.1f})"
        )

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
