# Multi-Agent Benchmark: GPT-OSS 20B vs Qwen3-Coder 30B vs Qwen3-Coder-Next 80B

**Date:** 2026-02-14
**Hardware:** RTX 4090 (24GB VRAM)
**Test:** 5 concurrent agents discussing a verifiable coding problem

---

## Test Configuration

| Model | Preset | VRAM | Concurrent Slots |
|-------|--------|------|------------------|
| GPT-OSS 20B MoE | `gpt-oss-20b-multi` | 14 GB | 5 |
| Qwen3-Coder 30B-A3B | `qwen3-coder-multi` | 20 GB | 5 |
| Qwen3-Coder-Next 80B MoE | `qwen3-coder-next-multi` | 22 GB | 5 |

---

## Agent Personas (5 Agents)

| Agent | System Prompt | Role |
|-------|---------------|------|
| **Architect** | "You are a software architect. Focus on design patterns, modularity, and maintainability." | High-level design |
| **Developer** | "You are a senior developer. Focus on implementation details and code correctness." | Implementation |
| **Tester** | "You are a QA engineer. Focus on edge cases, error handling, and test coverage." | Quality |
| **Security** | "You are a security engineer. Focus on vulnerabilities, input validation, and safe practices." | Security |
| **Reviewer** | "You are a code reviewer. Synthesize others' feedback and reach a final conclusion." | Synthesis |

---

## Test Task: Find the Bug

**Problem:** Given buggy code, agents must identify the bug and propose a fix.

### Code Sample (Python - Race Condition)

```python
import threading

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        current = self.count
        # Simulated delay
        self.count = current + 1

    def get_count(self):
        return self.count

def worker(counter, iterations):
    for _ in range(iterations):
        counter.increment()

# Test
counter = Counter()
threads = [threading.Thread(target=worker, args=(counter, 1000)) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Expected: 10000, Got: {counter.get_count()}")
```

**Expected Bug:** Race condition - non-atomic read-modify-write in `increment()`
**Expected Fix:** Use `threading.Lock()` or `+=` with lock

### Verification Criteria

| Criterion | Pass Condition |
|-----------|----------------|
| Bug Identified | Mentions "race condition" or "thread safety" or "non-atomic" |
| Root Cause | Identifies read-modify-write pattern |
| Fix Proposed | Suggests Lock, RLock, or atomic operation |
| Consensus | Final reviewer synthesizes and confirms |

---

## Metrics to Measure

### Latency Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| TTFT | Time to first token | Timestamp delta from request to first chunk |
| Response Time | Full response time per agent | Timestamp delta from request to completion |
| Total Time | Entire discussion time | Start to final consensus |

### Throughput Metrics

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| Tokens/sec (gen) | Generation speed | From llama.cpp `/health` or timing |
| Tokens/sec (prompt) | Prompt processing | From llama.cpp `/health` or timing |
| Concurrent Utilization | Active slots used | llama.cpp `/slots` endpoint |

### Quality Metrics

| Metric | Description | How to Score |
|--------|-------------|--------------|
| Correctness | Bug identified correctly | 0/1 binary |
| Fix Quality | Proposed fix is valid | 0-3 scale |
| Reasoning Quality | Clear explanation | 0-3 scale |
| Consensus Reached | Agents agree on answer | 0/1 binary |
| Turns to Consensus | Number of exchanges | Count |

---

## Test Protocol

### Phase 1: Individual Analysis (Parallel)

Send the code to all 5 agents simultaneously with their system prompts.

```bash
# All 5 requests sent at t=0
curl -X POST http://localhost:7070/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "[PERSONA PROMPT]"},
      {"role": "user", "content": "Analyze this code and identify any bugs:\n\n[CODE]"}
    ],
    "max_tokens": 512
  }'
```

### Phase 2: Discussion (Sequential)

Each agent responds to others' findings (2-3 rounds).

### Phase 3: Synthesis

Reviewer agent summarizes and declares consensus.

---

## Log Format

```json
{
  "model": "gpt-oss-20b-multi",
  "test_id": "20260214_001",
  "timestamp": "2026-02-14T01:30:00Z",
  "agents": [
    {
      "role": "architect",
      "ttft_ms": 245,
      "response_time_ms": 3200,
      "tokens_generated": 156,
      "tokens_per_sec": 48.7
    }
  ],
  "total_time_ms": 15000,
  "turns_to_consensus": 3,
  "correctness": {
    "bug_identified": true,
    "root_cause_correct": true,
    "fix_valid": true,
    "consensus_reached": true
  },
  "scores": {
    "correctness": 1,
    "fix_quality": 3,
    "reasoning_quality": 2
  }
}
```

---

## Test Script

```bash
#!/bin/bash
# multi_agent_test.sh

MODEL_PORT=7070
LOG_DIR="./benchmark_logs"
mkdir -p "$LOG_DIR"

# System prompts
ARCHITECT="You are a software architect. Focus on design patterns, modularity, and maintainability. Be concise."
DEVELOPER="You are a senior developer. Focus on implementation details and code correctness. Be concise."
TESTER="You are a QA engineer. Focus on edge cases, error handling, and test coverage. Be concise."
SECURITY="You are a security engineer. Focus on vulnerabilities, input validation, and safe practices. Be concise."
REVIEWER="You are a code reviewer. Synthesize others' feedback and reach a final conclusion. Be concise."

CODE='import threading

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        current = self.count
        self.count = current + 1

    def get_count(self):
        return self.count

def worker(counter, iterations):
    for _ in range(iterations):
        counter.increment()

counter = Counter()
threads = [threading.Thread(target=worker, args=(counter, 1000)) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Expected: 10000, Got: {counter.get_count()}")'

PROMPT="Analyze this code and identify any bugs. What is wrong and how would you fix it?\n\n\`\`\`python\n${CODE}\n\`\`\`"

# Function to send request and measure
send_request() {
    local role=$1
    local system_prompt=$2
    local start=$(date +%s%3N)

    response=$(curl -s -X POST "http://localhost:${MODEL_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [
                {\"role\": \"system\", \"content\": \"${system_prompt}\"},
                {\"role\": \"user\", \"content\": \"${PROMPT}\"}
            ],
            \"max_tokens\": 512,
            \"temperature\": 0.7
        }")

    local end=$(date +%s%3N)
    local duration=$((end - start))

    echo "{\"role\": \"${role}\", \"response_time_ms\": ${duration}, \"response\": ${response}}"
}

# Run test
echo "Starting multi-agent test at $(date)"
TEST_START=$(date +%s%3N)

# Phase 1: Send all 5 requests in parallel
echo "Phase 1: Individual Analysis (parallel)"
send_request "architect" "$ARCHITECT" > "$LOG_DIR/architect.json" &
send_request "developer" "$DEVELOPER" > "$LOG_DIR/developer.json" &
send_request "tester" "$TESTER" > "$LOG_DIR/tester.json" &
send_request "security" "$SECURITY" > "$LOG_DIR/security.json" &
send_request "reviewer" "$REVIEWER" > "$LOG_DIR/reviewer.json" &

wait

TEST_END=$(date +%s%3N)
TOTAL_TIME=$((TEST_END - TEST_START))

echo "Test complete. Total time: ${TOTAL_TIME}ms"
echo "Results saved to $LOG_DIR/"
```

---

## Expected Results Matrix

| Model | Est. TTFT | Est. tok/s | Est. Total Time | Notes |
|-------|-----------|------------|-----------------|-------|
| GPT-OSS 20B | ~200ms | ~30-50 | ~15-20s | Most VRAM headroom |
| Qwen3-Coder 30B | ~300ms | ~20-40 | ~20-30s | Good balance |
| Qwen3-Coder-Next 80B | ~500ms | ~5-15 | ~40-60s | CPU offload bottleneck |

---

## Procedure

1. **GPT-OSS 20B**
   ```bash
   gpumod switch multi-agent-gpt-oss
   ./multi_agent_test.sh
   ```

2. **Qwen3-Coder 30B**
   ```bash
   gpumod switch multi-agent-qwen3-coder
   ./multi_agent_test.sh
   ```

3. **Qwen3-Coder-Next 80B**
   ```bash
   gpumod switch multi-agent-qwen3-next
   ./multi_agent_test.sh
   ```

4. **Compare Results**
   - Parse JSON logs
   - Calculate averages
   - Score correctness
   - Generate comparison table

---

## Results

### GPT-OSS 20B MoE

| Agent | TTFT (ms) | Response Time (ms) | Tokens | tok/s |
|-------|-----------|-------------------|--------|-------|
| Architect | - | 7042 | 512 | 77 |
| Developer | - | 7033 | 512 | 77 |
| Tester | - | 7033 | 512 | 77 |
| Security | - | 7041 | 512 | 77 |
| Reviewer | - | 7032 | 512 | 77 |
| **Total** | - | **7057** | 2560 | 77 |

**Correctness:** [✗] Bug Identified [✗] Root Cause [✗] Fix Valid [✗] Consensus

**Note:** GPT-OSS 20B is a reasoning model. Analysis appears in `reasoning_content` field but `content` is empty. Needs higher `max_tokens` or different prompting to produce final answers.

---

### Qwen3-Coder 30B-A3B

| Agent | TTFT (ms) | Response Time (ms) | Tokens | tok/s |
|-------|-----------|-------------------|--------|-------|
| Architect | - | 5668 | 360 | 66 |
| Developer | - | 6142 | 387 | 65 |
| Tester | - | 6435 | 419 | 67 |
| Security | - | 6391 | 410 | 66 |
| Reviewer | - | 5303 | 341 | 67 |
| **Total** | - | **6451** | 1917 | 66 |

**Correctness:** [✓] Bug Identified [✓] Root Cause [✓] Fix Valid [✓] Consensus (5/5)

**Note:** All 5 agents correctly identified race condition, explained read-modify-write issue, and proposed Lock/RLock fix.

---

### Qwen3-Coder-Next 80B MoE

**FAILED TO LOAD:** Out of memory

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 25667.10 MiB on device 0: cudaMalloc failed: out of memory
```

The 80B model with 5 parallel slots and 40K context requires ~25.6GB VRAM, exceeding RTX 4090's 24GB.

**Options to fit:**
1. Reduce parallel slots from 5 to 3
2. Reduce context from 40K to 20K
3. Use CPU offloading (--n-gpu-layers 40 instead of -1)

---

## Summary Comparison (v1 - Simple Race Condition Bug)

| Metric | GPT-OSS 20B | Qwen3-Coder 30B | Qwen3-Next 80B |
|--------|-------------|-----------------|----------------|
| Total Time | 7.1s | **6.5s** | OOM |
| Avg tok/s | 77 | 66 | - |
| Total Tokens | 2560 | 1917 | - |
| Bug Found | 0/5 ❌ | **5/5** ✅ | - |
| VRAM Used | ~14GB | ~20GB | >25GB |
| Best For | Reasoning tasks | **Multi-agent coding** | Single-agent |

> **Note:** v1 showed GPT-OSS 0/5 because the benchmark didn't extract `reasoning_content`.
> See v2 below for corrected results.

---

## Benchmark v2 - Multiple Difficulty Levels

After fixing the reasoning model extraction issue and adding harder bug scenarios:

### Test Scenarios

| Difficulty | Bug Type | Description |
|------------|----------|-------------|
| Easy | Off-by-one | Pagination loses last page (integer division) |
| Medium | Resource Leak | Connection not released on exception |
| Hard | Async State | Shared mutable state across await boundaries |

### GPT-OSS 20B v2 Results

| Scenario | Bugs Found | Score | TTFT | Time |
|----------|------------|-------|------|------|
| Easy (Pagination) | **5/5** ✅ | 93.3% | 453ms | 14.0s |
| Medium (Resource Leak) | 4/5 | 66.7% | 471ms | 14.3s |
| Hard (Async State) | **5/5** ✅ | 93.3% | 676ms | 14.7s |

**Key Finding:** GPT-OSS works when extracting `reasoning_content` field!

### Qwen3-Coder 30B v2 Results

| Scenario | Bugs Found | Score | TTFT | Time |
|----------|------------|-------|------|------|
| Easy (Pagination) | 4/5 | 80.0% | 381ms | 5.6s |
| Medium (Resource Leak) | **5/5** ✅ | 86.7% | 291ms | 13.0s |
| Hard (Async State) | **5/5** ✅ | 86.7% | 602ms | 12.9s |

### v2 Summary Comparison

| Metric | GPT-OSS 20B | Qwen3-Coder 30B |
|--------|-------------|-----------------|
| Avg Total Time | 14.3s | 10.5s |
| Avg TTFT | 533ms | 425ms |
| Avg tok/s | 70 | 53 |
| Easy Score | **93.3%** | 80.0% |
| Medium Score | 66.7% | **86.7%** |
| Hard Score | **93.3%** | 86.7% |
| Avg Score | **84.4%** | 84.5% |

---

## Updated Conclusions

1. **Both models are now competitive** when properly configured:
   - GPT-OSS 20B: Excellent on easy and hard bugs (93.3%)
   - Qwen3-Coder 30B: Consistent across all difficulties (80-87%)

2. **Qwen3-Coder 30B is faster** (2x on easy, similar on complex):
   - Better TTFT (425ms vs 533ms)
   - Lower latency for interactive use

3. **GPT-OSS 20B requires `reasoning_content` extraction**:
   - Standard `content` field is empty
   - Analysis is in `reasoning_content` (like DeepSeek R1)
   - Higher token generation (70 tok/s) but longer responses

4. **Bug difficulty affects detection differently**:
   - GPT-OSS: Best on easy/hard, struggles on medium
   - Qwen3-Coder: Consistent performance, slightly better on medium

5. **Recommendation**:
   - **Qwen3-Coder 30B** for low-latency multi-agent workflows
   - **GPT-OSS 20B** for complex reasoning tasks with longer context

## Bug Found During Testing

See [gpumod-2p4]: Unit file not regenerated when preset config changes
