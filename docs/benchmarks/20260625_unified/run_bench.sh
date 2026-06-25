#!/usr/bin/env bash
# gpumod-kpmq.5 unified driver — Gemma 4 E2B (standard QAT UD-Q4_K_XL) +
# gemma4-26b-a4b-qat-mtp-q4, each with L1-L5 coding + AgentWorldBench (judge = claude -p).
# E2B is now the 4-bit standard tier (not mobile 2-bit) -> full coding, no token cap.
set +e
cd "$(git rev-parse --show-toplevel)"
OUT=docs/benchmarks/20260625_unified
SP="$(mktemp -d)"   # scratch for per-arm service-start logs
mkdir -p "$OUT"
start_wait() {
  uv run gpumod service start "$1" > "$SP/u4_start_$1.log" 2>&1 || { echo "!! START FAIL $1"; tail -6 "$SP/u4_start_$1.log"; return 1; }
  for i in $(seq 1 90); do curl -fsS "http://127.0.0.1:$2/health" >/dev/null 2>&1 && { echo "health OK $1 ~$((i*5))s"; return 0; }; sleep 5; done
  echo "!! HEALTH TIMEOUT $1"; return 1
}
run_pair() {  # $1=svc $2=port $3=model-key $4=coding-extra
  if start_wait "$1" "$2"; then
    echo "--- coding $3 $(date +%T) ($4) ---"
    uv run python scripts/run_qwen36_benchmark.py --model "$3" --output-dir "$OUT" $4 2>&1 | tail -4
    echo "--- agentworld $3 $(date +%T) ---"
    uv run python scripts/run_agentworld_benchmark.py --model-name "$3" --base-url "http://localhost:$2/v1" --sample 2 --output-dir "$OUT" 2>&1 | tail -8
  fi
  uv run gpumod service stop "$1" >/dev/null 2>&1; sleep 15
}
echo "### mode blank $(date +%T)"; uv run gpumod mode switch blank; sleep 15
echo "### E2B Q4 $(date +%T)"; run_pair gemma4-e2b-qat-q4 7113 gemma4-e2b-qat-q4 ""
echo "### MTP-Q4 $(date +%T)"; run_pair gemma4-26b-a4b-qat-mtp-q4 7110 gemma4-26b-a4b-qat-mtp-q4 ""
echo "### unified report $(date +%T)"
uv run python scripts/run_benchmarks.py \
  --coding "$OUT"/result_gemma4-e2b-qat-q4.json "$OUT"/result_gemma4-26b-a4b-qat-mtp-q4.json \
  --agentworld "$OUT"/result_agentworld_gemma4-e2b-qat-q4.json "$OUT"/result_agentworld_gemma4-26b-a4b-qat-mtp-q4.json \
  --output-dir "$OUT" \
  --title "Unified — Gemma 4 E2B (QAT Q4_K_XL) vs 26B-A4B-QAT-MTP (coding L1-L5 + AgentWorldBench, RTX 4090 b9784)" \
  --caveat "E2B = standard QAT UD-Q4_K_XL (4-bit, ~2.3B effective). Judge = Claude via 'claude -p' (opus), subscription, no API key." 2>&1 | tail -10
echo "### DONE $(date +%T)"; ls "$OUT"/result_*.json "$OUT"/README.md 2>/dev/null
