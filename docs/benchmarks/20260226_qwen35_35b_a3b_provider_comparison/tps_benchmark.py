#!/usr/bin/env python3
"""TPS and VRAM Benchmark for Qwen3.5-35B-A3B Provider Comparison.

Measures tokens per second and VRAM usage across different GGUF providers.

Usage:
    uv run python docs/benchmarks/20260226_qwen35_35b_a3b_provider_comparison/tps_benchmark.py \
        --model qwen35-35b-a3b-aessedai-iq4xs --port 7094 --runs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx


@dataclass
class TPSResult:
    """Result from a single TPS measurement run."""

    run: int
    prompt_tokens: int
    completion_tokens: int
    prompt_tps: float
    generation_tps: float
    total_time_s: float
    vram_mb: int


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model."""

    model_id: str
    port: int
    timestamp: str
    runs: list[TPSResult]
    vram_idle_mb: int
    vram_peak_mb: int

    @property
    def mean_generation_tps(self) -> float:
        return sum(r.generation_tps for r in self.runs) / len(self.runs)

    @property
    def std_generation_tps(self) -> float:
        mean = self.mean_generation_tps
        variance = sum((r.generation_tps - mean) ** 2 for r in self.runs) / len(self.runs)
        return variance ** 0.5


# Standard benchmark prompt (~500 tokens input)
BENCHMARK_PROMPT = """You are a senior software engineer. Analyze the following Python code and explain what it does, identify any bugs, and suggest improvements.

```python
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import defaultdict
import logging
import time
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    value: Any
    timestamp: float
    ttl: float
    hits: int = 0

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

class DistributedCache:
    def __init__(self, nodes: List[str], replication_factor: int = 2):
        self.nodes = nodes
        self.replication_factor = min(replication_factor, len(nodes))
        self.local_cache: Dict[str, CacheEntry] = {}
        self.stats = defaultdict(int)
        self._lock = asyncio.Lock()

    def _hash_key(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_nodes_for_key(self, key: str) -> List[str]:
        hash_val = self._hash_key(key)
        start_idx = hash_val % len(self.nodes)
        return [self.nodes[(start_idx + i) % len(self.nodes)]
                for i in range(self.replication_factor)]

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.local_cache:
                entry = self.local_cache[key]
                if not entry.is_expired():
                    entry.hits += 1
                    self.stats['hits'] += 1
                    return entry.value
                else:
                    del self.local_cache[key]
                    self.stats['expirations'] += 1

            self.stats['misses'] += 1
            return None

    async def set(self, key: str, value: Any, ttl: float = 300.0) -> bool:
        nodes = self._get_nodes_for_key(key)
        async with self._lock:
            self.local_cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )

        # Replicate to other nodes
        async with aiohttp.ClientSession() as session:
            tasks = []
            for node in nodes[1:]:
                tasks.append(self._replicate_to_node(session, node, key, value, ttl))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        return success_count >= self.replication_factor - 1

    async def _replicate_to_node(self, session, node: str, key: str,
                                  value: Any, ttl: float) -> bool:
        try:
            async with session.post(
                f"http://{node}/cache/set",
                json={"key": key, "value": value, "ttl": ttl},
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Replication to {node} failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        nodes = self._get_nodes_for_key(key)
        async with self._lock:
            self.local_cache.pop(key, None)
        return True

    def get_stats(self) -> Dict[str, int]:
        return dict(self.stats)
```

Provide a comprehensive analysis covering:
1. Overall architecture and design patterns used
2. Potential bugs or issues
3. Thread safety concerns
4. Performance optimizations
5. Suggested improvements"""


def get_vram_usage() -> int:
    """Get current VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def measure_tps(port: int, run_num: int, max_tokens: int = 200) -> TPSResult:
    """Run a single TPS measurement."""

    start_time = time.perf_counter()

    with httpx.Client(timeout=300.0) as client:
        response = client.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    total_time = time.perf_counter() - start_time

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Calculate TPS
    # Note: This is approximate - actual prompt processing time isn't separated
    prompt_tps = prompt_tokens / total_time if total_time > 0 else 0
    generation_tps = completion_tokens / total_time if total_time > 0 else 0

    vram = get_vram_usage()

    return TPSResult(
        run=run_num,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        total_time_s=total_time,
        vram_mb=vram,
    )


def run_benchmark(model_id: str, port: int, runs: int = 3) -> BenchmarkResult:
    """Run the complete TPS benchmark."""

    print(f"\n{'='*60}")
    print(f"TPS Benchmark: {model_id}")
    print(f"Port: {port}, Runs: {runs}")
    print("=" * 60)

    # Measure idle VRAM
    print("\nMeasuring idle VRAM...")
    time.sleep(2)
    vram_idle = get_vram_usage()
    print(f"  Idle VRAM: {vram_idle} MB")

    # Warmup run
    print("\nWarmup run...")
    try:
        _ = measure_tps(port, 0, max_tokens=50)
        print("  Warmup complete")
    except Exception as e:
        print(f"  Warmup failed: {e}")
        raise

    # Benchmark runs
    results: list[TPSResult] = []
    vram_peak = vram_idle

    for i in range(1, runs + 1):
        print(f"\nRun {i}/{runs}...")
        result = measure_tps(port, i)
        results.append(result)
        vram_peak = max(vram_peak, result.vram_mb)

        print(f"  Prompt tokens: {result.prompt_tokens}")
        print(f"  Completion tokens: {result.completion_tokens}")
        print(f"  Generation TPS: {result.generation_tps:.2f}")
        print(f"  Total time: {result.total_time_s:.2f}s")
        print(f"  VRAM: {result.vram_mb} MB")

    return BenchmarkResult(
        model_id=model_id,
        port=port,
        timestamp=datetime.now(UTC).isoformat(),
        runs=results,
        vram_idle_mb=vram_idle,
        vram_peak_mb=vram_peak,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TPS Benchmark")
    parser.add_argument("--model", required=True, help="Model preset ID")
    parser.add_argument("--port", type=int, required=True, help="LLM port")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")

    args = parser.parse_args()

    result = run_benchmark(args.model, args.port, args.runs)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {result.model_id}")
    print(f"Mean Generation TPS: {result.mean_generation_tps:.2f} ± {result.std_generation_tps:.2f}")
    print(f"VRAM Idle: {result.vram_idle_mb} MB")
    print(f"VRAM Peak: {result.vram_peak_mb} MB")

    # Save results
    output_file = args.output / f"tps_{args.model}.json"
    output_data = {
        "model_id": result.model_id,
        "port": result.port,
        "timestamp": result.timestamp,
        "vram_idle_mb": result.vram_idle_mb,
        "vram_peak_mb": result.vram_peak_mb,
        "mean_generation_tps": round(result.mean_generation_tps, 2),
        "std_generation_tps": round(result.std_generation_tps, 2),
        "runs": [
            {
                "run": r.run,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "generation_tps": round(r.generation_tps, 2),
                "total_time_s": round(r.total_time_s, 2),
                "vram_mb": r.vram_mb,
            }
            for r in result.runs
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
