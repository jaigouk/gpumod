"""Spike Q5: Probe vllm-metal and vllm-mlx endpoint compatibility.

Run AFTER installing vllm-metal or vllm-mlx in an isolated venv and starting
their servers. This script probes which endpoints exist and match VLLMDriver.

Usage:
    # Terminal 1: start vllm-mlx server
    uv pip install vllm-mlx
    python -m vllm_mlx.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8000

    # Terminal 2: run probe
    uv run python docs/internal/research/20260330_macos_spike/verify_vllm_endpoints.py --port 8000

    # Repeat with vllm-metal if available

Requires: A running vllm-metal or vllm-mlx server on localhost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install with: uv pip install httpx")
    sys.exit(1)


# Endpoints required by VLLMDriver (src/gpumod/services/drivers/vllm.py)
REQUIRED_ENDPOINTS = [
    # health_check (line 49-58)
    {"method": "GET", "path": "/health", "purpose": "health_check", "critical": True},
    # status — models list (line 60-80)
    {"method": "GET", "path": "/v1/models", "purpose": "status/model_info", "critical": True},
    # completions (line used by users)
    {"method": "POST", "path": "/v1/completions", "purpose": "inference",
     "body": {"model": "test", "prompt": "Hello", "max_tokens": 1}, "critical": True},
    {"method": "POST", "path": "/v1/chat/completions", "purpose": "chat_inference",
     "body": {"model": "test", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
     "critical": True},
    # sleep/wake (lines 85-160) — VRAM management
    {"method": "POST", "path": "/sleep", "purpose": "sleep (VRAM release)", "critical": False},
    {"method": "POST", "path": "/wake_up", "purpose": "wake (VRAM reclaim)", "critical": False},
    {"method": "GET", "path": "/is_sleeping", "purpose": "sleep_status", "critical": False},
]


def probe_endpoint(client: httpx.Client, base: str, ep: dict) -> dict:
    """Probe a single endpoint and return result."""
    url = f"{base}{ep['path']}"
    result = {
        "path": ep["path"],
        "method": ep["method"],
        "purpose": ep["purpose"],
        "critical": ep["critical"],
    }
    try:
        if ep["method"] == "GET":
            r = client.get(url, timeout=10)
        else:
            r = client.post(url, json=ep.get("body", {}), timeout=10)

        result["status_code"] = r.status_code
        result["available"] = r.status_code < 500
        result["response_preview"] = r.text[:200] if r.text else ""

        # Check for "not found" style responses
        if r.status_code == 404:
            result["available"] = False
            result["note"] = "404 Not Found — endpoint does not exist"
        elif r.status_code == 405:
            result["available"] = False
            result["note"] = "405 Method Not Allowed"

    except httpx.ConnectError:
        result["status_code"] = None
        result["available"] = False
        result["note"] = "Connection refused — server not running?"
    except httpx.TimeoutException:
        result["status_code"] = None
        result["available"] = False
        result["note"] = "Timeout"
    except Exception as e:
        result["status_code"] = None
        result["available"] = False
        result["note"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe vLLM endpoint compatibility")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-name", default="unknown", help="vllm-metal or vllm-mlx")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"=== Q5: vLLM endpoint compatibility probe ===")
    print(f"  Server: {args.server_name} at {base}")

    results: list[dict] = []
    critical_pass = 0
    critical_total = 0
    optional_pass = 0
    optional_total = 0

    with httpx.Client() as client:
        for ep in REQUIRED_ENDPOINTS:
            result = probe_endpoint(client, base, ep)
            results.append(result)

            status = "OK" if result["available"] else "MISSING"
            icon = "+" if result["available"] else "-"
            marker = " [CRITICAL]" if ep["critical"] else ""
            code = result.get("status_code") or "ERR"
            print(f"  [{icon}] {ep['method']:4} {ep['path']:25} → {code:>3} {status}{marker}")

            if ep["critical"]:
                critical_total += 1
                if result["available"]:
                    critical_pass += 1
            else:
                optional_total += 1
                if result["available"]:
                    optional_pass += 1

    # Summary
    print(f"\n  Critical endpoints: {critical_pass}/{critical_total}")
    print(f"  Optional endpoints: {optional_pass}/{optional_total} (sleep/wake)")

    compatible = critical_pass == critical_total
    has_sleep = optional_pass == optional_total
    print(f"\n  VLLMDriver compatible: {'YES' if compatible else 'NO'}")
    print(f"  Sleep/wake support:    {'YES' if has_sleep else 'NO'}")

    if compatible and has_sleep:
        verdict = "FULL — can use existing VLLMDriver as-is"
    elif compatible:
        verdict = "PARTIAL — basic VLLMDriver works, no VRAM management via sleep/wake"
    else:
        verdict = "INCOMPATIBLE — VLLMDriver cannot use this server"

    print(f"  Verdict: {verdict}")

    findings = {
        "server_name": args.server_name,
        "base_url": base,
        "endpoints": results,
        "critical_pass": critical_pass,
        "critical_total": critical_total,
        "optional_pass": optional_pass,
        "optional_total": optional_total,
        "compatible": compatible,
        "has_sleep_wake": has_sleep,
        "verdict": verdict,
    }

    out_path = f"docs/internal/research/20260330_macos_spike/vllm_{args.server_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
