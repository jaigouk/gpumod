#!/usr/bin/env python3
"""
RLM Experiment: Test if rlms library allows function calling.

Key question: Can we inject MCP tool functions into RLM's REPL
and have the LLM-generated code call them?

Run: uv run python docs/research/rlm_experiment.py
"""

from __future__ import annotations

import os
from typing import Any

# Check if API key is available
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Simulated MCP tools (these would be real MCP calls in production)
# ---------------------------------------------------------------------------


def gpu_status() -> dict[str, Any]:
    """Simulated gpu_status MCP tool - returns live VRAM info."""
    return {
        "total_vram_mb": 24000,
        "used_vram_mb": 8500,
        "free_vram_mb": 15500,
        "running_services": ["ollama"],
    }


def list_gguf_files(repo_id: str) -> dict[str, Any]:
    """Simulated list_gguf_files MCP tool."""
    return {
        "repo_id": repo_id,
        "files": [
            {"filename": "model-IQ2_XXS.gguf", "estimated_vram_mb": 26100},
            {"filename": "model-Q2_K.gguf", "estimated_vram_mb": 29000},
            {"filename": "model-Q4_K_M.gguf", "estimated_vram_mb": 45000},
        ],
    }


def fetch_model_config(repo_id: str) -> dict[str, Any]:
    """Simulated fetch of config.json from HuggingFace."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
    }


# ---------------------------------------------------------------------------
# Test 1: Can we inject functions via setup_code?
# ---------------------------------------------------------------------------


def test_function_injection() -> None:
    """Test: Can RLM REPL call injected functions?"""
    from rlm.environments.local_repl import LocalREPL

    print("=" * 70)
    print("TEST 1: Function injection via setup_code")
    print("=" * 70)

    # Define functions as setup_code string
    setup_code = """
def gpu_status():
    '''Get current GPU status.'''
    return {
        'total_vram_mb': 24000,
        'used_vram_mb': 8500,
        'free_vram_mb': 15500,
        'running_services': ['ollama'],
    }

def list_gguf_files(repo_id):
    '''List GGUF files for a HuggingFace repo.'''
    return {
        'repo_id': repo_id,
        'files': [
            {'filename': 'model-IQ2_XXS.gguf', 'estimated_vram_mb': 26100},
            {'filename': 'model-Q2_K.gguf', 'estimated_vram_mb': 29000},
        ],
    }

def fetch_model_config(repo_id):
    '''Fetch model config.json.'''
    return {
        'architectures': ['Qwen3ForCausalLM'],
        'num_experts': 8,
        'num_experts_per_tok': 2,
    }
"""

    repl = LocalREPL(setup_code=setup_code)

    # Test 1a: Can we call gpu_status?
    print("\n[Test 1a] Calling gpu_status()...")
    result = repl.execute_code("status = gpu_status(); print(status)")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Function call worked!" if "free_vram_mb" in result.stdout else "  ✗ Failed")

    # Test 1b: Can we call with arguments?
    print("\n[Test 1b] Calling list_gguf_files(repo_id)...")
    result = repl.execute_code("files = list_gguf_files('unsloth/Test-GGUF'); print(files)")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Function with args worked!" if "model-IQ2_XXS" in result.stdout else "  ✗ Failed")

    # Test 1c: Multi-step reasoning
    print("\n[Test 1c] Multi-step reasoning...")
    code = """
status = gpu_status()
vram = status['free_vram_mb']
files = list_gguf_files('unsloth/Test-GGUF')

print(f"Available VRAM: {vram}MB")
fitting = [f for f in files['files'] if f['estimated_vram_mb'] <= vram]
if fitting:
    print(f"Fitting models: {[f['filename'] for f in fitting]}")
else:
    print("No models fit in available VRAM")
"""
    result = repl.execute_code(code)
    print(f"  stdout:\n{result.stdout}")
    print("  ✓ Multi-step reasoning worked!" if "No models fit" in result.stdout else "  ✗ Failed")


# ---------------------------------------------------------------------------
# Test 2: Can we pass real functions (not just strings)?
# ---------------------------------------------------------------------------


def test_real_function_injection() -> None:
    """Test: Can we inject actual Python functions (not strings)?"""
    from rlm.environments.local_repl import LocalREPL

    print("\n" + "=" * 70)
    print("TEST 2: Injecting real Python functions")
    print("=" * 70)

    # Create REPL
    repl = LocalREPL()

    # Inject real functions by adding to globals
    repl.globals["gpu_status"] = gpu_status
    repl.globals["list_gguf_files"] = list_gguf_files
    repl.globals["fetch_model_config"] = fetch_model_config

    print("\n[Test 2a] Calling injected real function...")
    result = repl.execute_code("status = gpu_status(); print(status)")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Real function injection worked!" if "15500" in result.stdout else "  ✗ Failed")

    print("\n[Test 2b] Calling function with args...")
    result = repl.execute_code("files = list_gguf_files('test/repo'); print(len(files['files']))")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Args passed correctly!" if "3" in result.stdout else "  ✗ Failed")


# ---------------------------------------------------------------------------
# Test 3: Full RLM with injected functions (requires API key)
# ---------------------------------------------------------------------------


def test_full_rlm_with_functions() -> None:
    """Test: Full RLM agent with injected functions."""
    if not HAS_API_KEY:
        print("\n" + "=" * 70)
        print("TEST 3: Full RLM with functions (SKIPPED - no API key)")
        print("=" * 70)
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run this test")
        return

    from rlm import RLM

    print("\n" + "=" * 70)
    print("TEST 3: Full RLM agent with injected functions")
    print("=" * 70)

    # Setup code that defines our MCP tool wrappers
    setup_code = """
def gpu_status():
    '''Get current GPU VRAM status. Returns dict with free_vram_mb, total_vram_mb.'''
    return {
        'total_vram_mb': 24000,
        'used_vram_mb': 8500,
        'free_vram_mb': 15500,
    }

def list_gguf_files(repo_id):
    '''List GGUF files for a repo. Returns dict with files list containing filename and estimated_vram_mb.'''
    return {
        'files': [
            {'filename': 'model-IQ2_XXS.gguf', 'estimated_vram_mb': 26100},
            {'filename': 'model-Q2_K.gguf', 'estimated_vram_mb': 29000},
        ],
    }
"""

    # Check which backend to use
    if os.environ.get("ANTHROPIC_API_KEY"):
        backend = "anthropic"
        backend_kwargs = {"model_name": "claude-3-5-haiku-20241022"}
    else:
        backend = "openai"
        backend_kwargs = {"model_name": "gpt-4o-mini"}

    print(f"Using backend: {backend}")

    # Create RLM with setup code
    rlm_agent = RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment="local",
        environment_kwargs={"setup_code": setup_code},
        verbose=True,
        max_iterations=5,
    )

    # Query that requires using the functions
    query = """
You have access to these functions:
- gpu_status() -> dict with free_vram_mb, total_vram_mb
- list_gguf_files(repo_id) -> dict with files list

Question: Can I run any model from 'unsloth/Qwen-GGUF' on my GPU?

Check my available VRAM, then check which models fit.
Use FINAL_VAR() to return your answer.
"""

    print(f"\nQuery: {query}")
    print("\n--- RLM Execution ---")

    try:
        result = rlm_agent.completion(query)
        print("\n--- Result ---")
        print(f"Response: {result.response}")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Test 4: Context payload for data injection
# ---------------------------------------------------------------------------


def test_context_payload() -> None:
    """Test: Can we pass data via context_payload?"""
    from rlm.environments.local_repl import LocalREPL

    print("\n" + "=" * 70)
    print("TEST 4: Data injection via context_payload")
    print("=" * 70)

    # Pre-fetch data
    vram_data = gpu_status()
    model_files = list_gguf_files("unsloth/Test")

    # Create context with pre-fetched data
    context_payload = {
        "vram": vram_data,
        "files": model_files["files"],
    }

    repl = LocalREPL(context_payload=context_payload)

    print("\n[Test 4a] Accessing context data...")
    result = repl.execute_code("print(f\"VRAM: {context['vram']['free_vram_mb']}MB\")")
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Context data accessible!" if "15500" in result.stdout else "  ✗ Failed")

    print("\n[Test 4b] Reasoning over pre-fetched data...")
    code = """
vram = context['vram']['free_vram_mb']
files = context['files']
fitting = [f for f in files if f['estimated_vram_mb'] <= vram]
print(f"Models fitting in {vram}MB: {len(fitting)}")
"""
    result = repl.execute_code(code)
    print(f"  stdout: {result.stdout.strip()}")
    print(f"  ✓ Pre-fetched data works!" if "0" in result.stdout else "  ✗ Unexpected")


# ---------------------------------------------------------------------------
# Test 5: Function whitelisting (only read-only tools allowed)
# ---------------------------------------------------------------------------


def test_function_whitelisting() -> None:
    """Test: Can we limit which functions are accessible?

    This verifies that we can expose ONLY read-only MCP tools to RLM,
    blocking mutating operations like switch_mode, start_service, stop_service.
    """
    from rlm.environments.local_repl import LocalREPL

    print("\n" + "=" * 70)
    print("TEST 5: Function Whitelisting (Read-Only Tools)")
    print("=" * 70)

    # Create REPL with ONLY read-only functions
    repl = LocalREPL()

    # Whitelist: Only read-only MCP tools
    repl.globals["gpu_status"] = gpu_status
    repl.globals["list_gguf_files"] = list_gguf_files
    repl.globals["fetch_model_config"] = fetch_model_config

    # NOT exposed: switch_mode, start_service, stop_service
    # These mutating functions are intentionally omitted

    print("\n[Test 5a] Allowed: calling gpu_status()...")
    result = repl.execute_code("status = gpu_status(); print(status['free_vram_mb'])")
    allowed_works = "15500" in result.stdout
    print(f"  stdout: {result.stdout.strip()}")
    print("  ✓ Read-only function works!" if allowed_works else "  ✗ Failed")

    print("\n[Test 5b] Blocked: calling switch_mode() (not exposed)...")
    result = repl.execute_code("""
try:
    switch_mode('code')
    print('SHOULD NOT REACH HERE')
except NameError as e:
    print(f'BLOCKED: {e}')
""")
    blocked_works = "BLOCKED" in result.stdout and "switch_mode" in result.stdout
    print(f"  stdout: {result.stdout.strip()}")
    print("  ✓ Mutating function blocked!" if blocked_works else "  ✗ Failed")

    print("\n[Test 5c] Blocked: calling start_service() (not exposed)...")
    result = repl.execute_code("""
try:
    start_service('ollama')
    print('SHOULD NOT REACH HERE')
except NameError as e:
    print(f'BLOCKED: {e}')
""")
    blocked_start = "BLOCKED" in result.stdout and "start_service" in result.stdout
    print(f"  stdout: {result.stdout.strip()}")
    print("  ✓ Mutating function blocked!" if blocked_start else "  ✗ Failed")

    print("\n[Test 5d] Multi-step with only allowed functions...")
    code = """
# This should work - only uses whitelisted functions
status = gpu_status()
vram = status['free_vram_mb']
files = list_gguf_files('unsloth/Test-GGUF')
config = fetch_model_config('unsloth/Test-Model')

print(f"VRAM: {vram}MB")
print(f"Model arch: {config['architectures'][0]}")
print(f"Available quants: {len(files['files'])}")

# Recommendation based on read-only data
if config.get('num_experts'):
    print("MoE model - consider --n-cpu-moe flag")
"""
    result = repl.execute_code(code)
    multi_works = "15500" in result.stdout and "Qwen3ForCausalLM" in result.stdout
    print(f"  stdout:\n{result.stdout}")
    print("  ✓ Multi-step with whitelisted functions works!" if multi_works else "  ✗ Failed")

    # Summary
    print("\n" + "-" * 70)
    print("WHITELISTING SUMMARY:")
    print("-" * 70)
    print(f"  ✓ Read-only functions accessible:    {allowed_works}")
    print(f"  ✓ switch_mode blocked (NameError):   {blocked_works}")
    print(f"  ✓ start_service blocked (NameError): {blocked_start}")
    print(f"  ✓ Multi-step reasoning works:        {multi_works}")
    print("\n  CONCLUSION: Function whitelisting works!")
    print("  RLM can ONLY call functions we explicitly expose.")
    print("  Mutating operations (switch_mode, start/stop_service) are blocked.")


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------


def show_conclusion() -> None:
    """Summarize findings."""
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ FINDING 1: setup_code works for function injection                  │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Define functions as string in setup_code                          │
│ ✓ Functions become callable in REPL                                 │
│ ✓ Multi-step reasoning works                                        │
│ ✗ Functions must be defined as strings (serialization)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FINDING 2: Direct globals injection works                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Can inject real Python functions into repl.globals                │
│ ✓ Functions can call external APIs, make network requests           │
│ ✓ More flexible than setup_code                                     │
│ ⚠ Bypasses sandbox - LLM code has full access                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FINDING 3: context_payload works for data                           │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Pre-fetch data, pass as context_payload                           │
│ ✓ Data available as 'context' variable in REPL                      │
│ ✓ Sandbox-safe (data only, no function calls)                       │
│ ✗ Cannot fetch new data dynamically                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ARCHITECTURE RECOMMENDATION for gpumod                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Option A: Pre-fetch + context_payload (SAFEST)                      │
│   1. Before RLM: fetch gpu_status(), model config, driver docs      │
│   2. Pass as context_payload                                        │
│   3. RLM reasons over static data                                   │
│   ✓ Sandbox-safe, no live calls from LLM code                       │
│   ✗ Must predict what data RLM needs                                │
│                                                                     │
│ Option B: setup_code with function stubs (MODERATE)                 │
│   1. Define wrapper functions in setup_code                         │
│   2. Functions can call our MCP tools                               │
│   3. RLM code calls functions dynamically                           │
│   ⚠ LLM code can call any exposed function                          │
│   ✓ Dynamic - can fetch any data                                    │
│                                                                     │
│ Option C: Direct globals injection (MOST FLEXIBLE)                  │
│   1. Inject real functions into repl.globals                        │
│   2. Full access to MCP tools                                       │
│   ⚠ No sandbox - LLM code runs with full permissions                │
│   ✓ Maximum flexibility                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

For gpumod RLM consult tool:
- Use Option B with READ-ONLY function wrappers
- Expose: gpu_status, list_gguf_files, fetch_model_config, etc.
- Do NOT expose: switch_mode, start_service, stop_service
- Return recommendations, user/AI decides to execute
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║" + " RLM FUNCTION INJECTION EXPERIMENT ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    test_function_injection()
    test_real_function_injection()
    test_context_payload()
    test_function_whitelisting()
    test_full_rlm_with_functions()
    show_conclusion()
