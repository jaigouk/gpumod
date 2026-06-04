#!/usr/bin/env bash
# Central coding-benchmark driver — replaces per-bench-dir scripts with one
# parameterised invocation. See gpumod-4omn for design notes.
#
# Usage:
#   scripts/run_coding_benchmark.sh --output-dir DIR [options] MODEL [MODEL ...]
#
# Examples:
#   scripts/run_coding_benchmark.sh \
#       --output-dir docs/benchmarks/20260603_gemma4_12b_vs_qwen36_35b_a3b_mtp/ \
#       gemma4-12b-q4 gemma4-12b-q5 gemma4-12b-q8 gemma4-26b-a4b-q4
#
#   scripts/run_coding_benchmark.sh \
#       --output-dir docs/benchmarks/<dir>/ \
#       --size-guard 12288:$HOME/bin/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
#       gemma4-26b-a4b-q4
#
#   scripts/run_coding_benchmark.sh \
#       --output-dir docs/benchmarks/<dir>/ \
#       --no-archive gemma4-12b-q4
#
# Launch inside tmux per .claude/CLAUDE.md "Running Long Benchmarks".
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# --- defaults ---
OUTPUT_DIR=""
ARCHIVE=true
MODE_SWITCH=true
QUIESCE_SECS=20
RUNNER="scripts/run_qwen36_benchmark.py"
declare -a SIZE_GUARDS=()
declare -a MODELS=()

usage() {
    cat <<EOF
Usage: $0 [options] MODEL [MODEL ...]

Options:
  --output-dir DIR        (required) directory to write results into
  --no-archive            skip archive_prior() — use for genuine first runs
  --no-mode-switch        skip the 'gpumod mode switch blank' VRAM-isolation guard
  --quiesce SEC           seconds to wait between models (default: ${QUIESCE_SECS})
  --runner PATH           python runner script (default: ${RUNNER})
  --size-guard MIB:PATH   refuse if PATH is smaller than MIB MiB (repeatable)
  --help, -h              this message

Per-model behaviour:
  1. archive_prior (if --no-archive not set): if result_<model>.json exists,
     rename it + run_<model>.log + artifacts/<model>/ to the next free .runN slot.
  2. gpumod service start <model>
  3. wait for /health on the port (read from presets/llm/<model>.yaml)
  4. uv run python <runner> --model <model> --output-dir <DIR>
  5. gpumod service stop <model>
  6. sleep <quiesce>
EOF
}

# --- arg parsing ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --no-archive)        ARCHIVE=false; shift ;;
        --no-mode-switch)    MODE_SWITCH=false; shift ;;
        --quiesce)           QUIESCE_SECS="$2"; shift 2 ;;
        --runner)            RUNNER="$2"; shift 2 ;;
        --size-guard)        SIZE_GUARDS+=("$2"); shift 2 ;;
        --help|-h)           usage; exit 0 ;;
        --*)                 echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)                   MODELS+=("$1"); shift ;;
    esac
done

# --- validation ---
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output-dir is required" >&2
    usage >&2
    exit 2
fi
if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: at least one MODEL is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output-dir does not exist: $OUTPUT_DIR" >&2
    exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: --runner does not exist: $RUNNER" >&2
    exit 2
fi

# --- helpers ---
preset_port() {
    local model="$1"
    local preset="presets/llm/${model}.yaml"
    if [[ ! -f "$preset" ]]; then
        echo "ERROR: preset not found: $preset" >&2
        return 1
    fi
    python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['port'])" "$preset"
}

archive_prior() {
    local model="$1"
    if [[ ! -f "$OUTPUT_DIR/result_${model}.json" ]]; then
        return 0
    fi
    local n=1
    while [[ -e "$OUTPUT_DIR/result_${model}.run${n}.json" \
        || -e "$OUTPUT_DIR/run_${model}.run${n}.log" \
        || -e "$OUTPUT_DIR/artifacts/${model}.run${n}" ]]; do
        n=$((n + 1))
    done
    echo "=== $(date -Iseconds) archiving current ${model} data → run${n} ==="
    mv "$OUTPUT_DIR/result_${model}.json" "$OUTPUT_DIR/result_${model}.run${n}.json"
    if [[ -f "$OUTPUT_DIR/run_${model}.log" ]]; then
        mv "$OUTPUT_DIR/run_${model}.log" "$OUTPUT_DIR/run_${model}.run${n}.log"
    fi
    if [[ -d "$OUTPUT_DIR/artifacts/${model}" ]]; then
        mv "$OUTPUT_DIR/artifacts/${model}" "$OUTPUT_DIR/artifacts/${model}.run${n}"
    fi
}

check_size_guards() {
    local spec mib path size_bytes min_bytes actual_mib
    for spec in "${SIZE_GUARDS[@]}"; do
        mib="${spec%%:*}"
        path="${spec#*:}"
        # Expand a leading $HOME or ~ (the shell did not expand inside the arg).
        path="${path//\$HOME/$HOME}"
        path="${path/#\~/$HOME}"
        if [[ ! -f "$path" ]]; then
            echo "ERROR: size-guard file not found: $path" >&2
            exit 1
        fi
        size_bytes=$(stat -c%s "$path")
        min_bytes=$((mib * 1024 * 1024))
        if [[ "$size_bytes" -lt "$min_bytes" ]]; then
            actual_mib=$((size_bytes / 1024 / 1024))
            echo "ERROR: $path is only ${actual_mib} MiB — expected ${mib}+ MiB" >&2
            exit 1
        fi
    done
}

run_one() {
    local model="$1" port="$2"
    if [[ "$ARCHIVE" == true ]]; then
        archive_prior "$model"
    fi
    echo "=== $(date -Iseconds) starting service ${model} on port ${port} ==="
    uv run gpumod service start "$model"
    echo "=== waiting for /health on ${port} ==="
    local waited=0
    until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
        sleep 3
        waited=$((waited + 3))
        if [[ "$waited" -gt 300 ]]; then
            echo "!!! ${model} failed to come up in 5min — aborting" >&2
            uv run gpumod service stop "$model" || true
            exit 1
        fi
    done
    echo "=== health OK after ${waited}s, launching benchmark ==="
    uv run python "$RUNNER" \
        --model "$model" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/run_${model}.log"
    echo "=== $(date -Iseconds) benchmark ${model} done, stopping service ==="
    uv run gpumod service stop "$model"
    echo "=== quiescing ${QUIESCE_SECS}s before next model ==="
    sleep "$QUIESCE_SECS"
}

# --- main ---
check_size_guards

if [[ "$MODE_SWITCH" == true ]]; then
    echo "=== $(date -Iseconds) ensuring blank mode (VRAM isolation) ==="
    uv run gpumod mode switch blank
fi

for model in "${MODELS[@]}"; do
    port=$(preset_port "$model")
    run_one "$model" "$port"
done

echo "=== $(date -Iseconds) ALL BENCHMARKS COMPLETE ==="
