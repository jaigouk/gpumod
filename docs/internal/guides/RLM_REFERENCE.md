# RLM (Recursive Language Model) Reference Guide

> A practical guide for developers who want to understand and implement RLM patterns for complex reasoning tasks.

**What you'll learn:**
- Why traditional approaches fail on complex, multi-step questions
- How RLM treats context as a variable to explore programmatically
- The REPL loop pattern that powers iterative reasoning
- Security considerations for executing LLM-generated code
- Patterns for building your own RLM-powered applications

**Prerequisites:**
- Basic Python knowledge
- Familiarity with LLM APIs (OpenAI, Anthropic)
- No prior RLM experience required

---

## Table of Contents

1. [The Problem: Why Traditional RAG Falls Short](#1-the-problem-why-traditional-rag-falls-short)
2. [The Solution: Context as a Variable](#2-the-solution-context-as-a-variable)
3. [Core Concepts](#3-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Security: Sandboxing Code Execution](#5-security-sandboxing-code-execution)
6. [Building Your Own RLM Application](#6-building-your-own-rlm-application)
7. [Implementation Patterns](#7-implementation-patterns)
8. [Common Pitfalls and Solutions](#8-common-pitfalls-and-solutions)
9. [References](#9-references)

---

## 1. The Problem: Why Traditional RAG Falls Short

### 1.1 The Context Window Problem

Consider any question that requires reasoning across multiple knowledge sources:

- "Can I run Model X on my hardware?" (requires: model specs, hardware limits, optimization options)
- "Is this contract compliant with regulation Y?" (requires: contract text, regulation details, cross-references)
- "Why is my application slow?" (requires: logs, metrics, documentation, config files)

**Option A: Stuff everything into the prompt**
```python
response = llm.complete(f"""
Here's all the documentation:
{doc_1}  # 50KB
{doc_2}  # 30KB
{doc_3}  # 20KB

Question: {user_question}
""")
```

Problems:
- Context windows have limits (even 128K tokens isn't infinite)
- Cost scales linearly with input size
- The LLM may "lose focus" in long contexts
- You're paying for everything even if only parts are relevant

**Option B: Traditional RAG (Retrieval-Augmented Generation)**
```python
# 1. Chunk the documentation
chunks = split_into_chunks(all_docs, size=500)

# 2. Embed and store
for chunk in chunks:
    vector_db.add(embed(chunk))

# 3. Retrieve relevant chunks
relevant = vector_db.search(user_question, top_k=5)

# 4. Generate answer from fragments
response = llm.complete(f"Based on these excerpts: {relevant}\n\nAnswer: ...")
```

This is better, but has fundamental limitations.

### 1.2 The Dependency Chain Problem

Complex questions have **dependency chains** — answering one part reveals what you need to look up next:

```
"Can I run Model X on 24GB VRAM?"
    │
    ├─→ What's the model architecture? ─→ It's MoE (Mixture of Experts)
    │                                        │
    │                                        └─→ Can I offload parts to CPU?
    │                                               │
    │                                               └─→ Check driver documentation
    │
    ├─→ What sizes are available?
    │       │
    │       └─→ List available files → Smallest is 26GB
    │                                    │
    │                                    └─→ With offloading? Maybe fits
    │
    └─→ What's my actual available memory?
            │
            └─→ Check system status → 22GB free
```

Traditional RAG retrieves fragments in isolation. It might get the model specs but miss:
- The architecture detail that enables optimization
- The driver flag that makes it possible
- The calculation formula

**The fundamental issue:** RAG treats knowledge as passive data to be searched. But complex analysis requires *active exploration* — following leads, checking cross-references, and building cumulative understanding.

### 1.3 The Needle in a Haystack

Technical documentation has "needles" — critical details buried in dense text:

- A configuration flag that enables a key optimization
- An edge case mentioned in a footnote
- A version-specific feature or limitation

RAG's similarity search might not surface these if the query doesn't match the embedding well enough.

---

## 2. The Solution: Context as a Variable

### 2.1 The Key Insight

> Instead of reading information linearly, we treat it as a **variable in a Python environment**. The LLM acts as a reasoning engine that writes code to explore that variable.

This is the core insight from the RLM paper (Zhang et al., 2025).

**Traditional approach:**
```python
# Context goes INTO the prompt
llm.complete(prompt + context)
```

**RLM approach:**
```python
# Context is a VARIABLE the LLM can explore via code
context = {"query": user_question, "data": initial_data}

# LLM writes code to explore it
rlm.complete("Answer the user's query")
# -> LLM generates: result = search_database(query)
# -> LLM generates: details = fetch_details(result['id'])
# -> LLM generates: related = find_related(details)
# -> etc.
```

### 2.2 The Detective Analogy

Think of the LLM as a detective investigating a case:

**Traditional RAG** = Detective receives a folder with 5 random pages from a 500-page case file. "Here's what seemed relevant. Good luck!"

**RLM** = Detective has access to tools and databases. They can:
- Query databases and APIs
- Check system status
- Fetch documentation
- Cross-reference findings
- Ask a colleague (sub-query) to analyze a specific piece

The detective decides what to investigate next based on what they've found so far.

### 2.3 What Makes RLM Different

| Aspect | Traditional RAG | RLM |
|--------|-----------------|-----|
| Context handling | Fragments retrieved upfront | Explored on-demand via code |
| Navigation | Static (similarity search) | Dynamic (code-driven) |
| Dependencies | Often missed | Can follow chains |
| Iteration | Single retrieval step | Multiple exploration steps |
| LLM role | Answer generator | Reasoning engine + code writer |

---

## 3. Core Concepts

### 3.1 The REPL Loop

REPL stands for **Read-Eval-Print Loop** — the interactive programming pattern where you:
1. **Read** input (in RLM: the LLM's generated code)
2. **Eval**uate (execute) that code
3. **Print** the result
4. **Loop** back for more input

In RLM, the loop looks like this:

```
┌─────────────────────────────────────────────────────────────┐
│                        RLM REPL Loop                        │
└─────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │  User Query  │
     └──────┬───────┘
            │
            ▼
┌───────────────────────┐
│   LLM generates code  │◄─────────────────────────┐
│   in ```repl``` block │                          │
└───────────┬───────────┘                          │
            │                                      │
            ▼                                      │
┌───────────────────────┐                          │
│   Execute code in     │                          │
│   sandboxed REPL      │                          │
└───────────┬───────────┘                          │
            │                                      │
            ▼                                      │
┌───────────────────────┐                          │
│   Capture stdout/     │──────────────────────────┤
│   stderr output       │                          │
└───────────┬───────────┘                          │
            │                                      │
            ▼                                      │
┌───────────────────────┐                          │
│   Feed output back    │                          │
│   to LLM as context   │──────────────────────────┘
└───────────┬───────────┘
            │
            ▼ (when LLM outputs FINAL(...))
┌───────────────────────┐
│   Return structured   │
│   answer to user      │
└───────────────────────┘
```

### 3.2 Code Blocks

The LLM writes code in specially marked blocks:

````markdown
I'll start by examining the data.

```repl
# Check what we're working with
print(f"Query: {context['query']}")
result = search_database(context['query'])
print(f"Found {len(result)} matches")
print(result[:3])  # Preview first 3
```
````

The system:
1. Extracts code between ` ```repl ` and ` ``` `
2. Executes it in a sandboxed environment
3. Captures `print()` output
4. Sends output back to the LLM

### 3.3 The FINAL Answer Pattern

When the LLM has gathered enough information, it signals completion:

```python
FINAL({
    "answer": "Yes, this is possible with configuration X",
    "recommendation": "Use option A with setting B for best results",
    "reasoning_steps": [
        "First, I checked the requirements",
        "Then, I verified the constraints",
        "Finally, I found a compatible configuration"
    ],
    "sources": ["search_database", "fetch_config", "check_status"]
})
```

The `FINAL(...)` marker tells the system to stop the loop and parse the answer.

### 3.4 Recursive Sub-Queries (llm_query)

Sometimes the LLM needs focused analysis on a specific piece:

```python
```repl
# This section is complex. Let me analyze it separately.
detailed_analysis = llm_query(
    "What are the key constraints in this configuration?",
    config_text
)
print(detailed_analysis)
```
```

The `llm_query()` function spawns a sub-RLM call with a focused prompt. This is the "recursive" in Recursive Language Model — the system can call itself to analyze sub-problems.

**Why use sub-queries?**
- Keeps the main conversation focused
- Allows deeper analysis without polluting the main context
- Can use different parameters (e.g., a smaller/cheaper model for simple checks)

### 3.5 Available Tools

Beyond raw code execution, RLM environments expose domain-specific tool functions:

```python
# Example tools in the REPL namespace
search_database(query)       # Search your data
fetch_details(id)            # Get detailed information
check_status()               # Check system/resource state
fetch_documentation(topic)   # Get relevant docs
llm_query(prompt, context)   # Recursive sub-call
```

**Key principle:** Only expose **read-only tools**. The RLM provides recommendations but never executes mutations (modifying data, starting processes, etc.).

---

## 4. Architecture Deep Dive

### 4.1 System Prompt Anatomy

The system prompt teaches the LLM how to use the REPL:

```python
SYSTEM_PROMPT = """
You are an assistant operating inside a REPL environment.

## Available Functions

Call these directly in ```repl``` code blocks:

| Function               | Description                          |
|------------------------|--------------------------------------|
| `search(query)`        | Search the knowledge base            |
| `fetch(id)`            | Get detailed information by ID       |
| `status()`             | Check current system state           |
| `docs(topic)`          | Fetch documentation on a topic       |

You also have `json`, `re`, and `math` available as modules.

## Rules

1. **Explore first.** Always call at least one function before answering.
2. **Cite sources.** Reference which function call provided each fact.
3. **No imports.** Do not use `import`, `exec`, `eval`, or `open`.
4. **No mutations.** You only provide recommendations; the user decides.
5. **Be concise.** Keep code blocks short and focused.
6. **Use print().** Print intermediate results so you can reason over them.

## Workflow

1. Examine the `context` variable (your query + any pre-loaded data).
2. Call tool functions to gather data.
3. Reason over results step-by-step.
4. When ready, provide your final answer using:

FINAL({
  "answer": "...",
  "reasoning_steps": ["step 1", "step 2"],
  "sources": ["source 1", "source 2"]
})
"""
```

**Key elements:**
- **Available functions table** — what the LLM can call
- **Rules** — constraints that guide safe behavior
- **Workflow** — the expected exploration pattern
- **FINAL format** — how to signal completion

### 4.2 The Orchestrator

The orchestrator manages the RLM lifecycle:

```python
class RLMOrchestrator:
    def consult(self, query: str, max_turns: int = 5) -> Result:
        # 1. Create sandboxed environment
        env = SafeREPLEnv(tool_wrappers=self._tools)

        # 2. Initialize message history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Answer: {query}"}
        ]

        # 3. Run the REPL loop
        for turn in range(max_turns):
            response = llm.complete(messages)

            # Check for final answer
            if "FINAL(" in response:
                return parse_final_answer(response)

            # Extract and execute code blocks
            code_blocks = find_code_blocks(response)
            for code in code_blocks:
                output = env.execute_code(code)
                messages.append({"role": "user", "content": f"Output: {output}"})

        # 4. Force final answer if max turns reached
        return force_final_answer(messages)
```

### 4.3 The Environment

The environment provides:
1. **Namespace isolation** — code runs in a restricted scope
2. **Tool injection** — whitelisted functions are available
3. **Safe builtins** — only safe Python builtins allowed
4. **Output capture** — stdout/stderr are captured and returned

```python
class SafeREPLEnv:
    WHITELISTED_TOOLS = frozenset({
        "search",
        "fetch",
        "status",
        "docs",
    })

    BLOCKED_TOOLS = frozenset({
        "delete",      # Mutating!
        "update",      # Mutating!
        "execute",     # Mutating!
    })

    def execute_code(self, code: str) -> REPLResult:
        # 1. Validate code via AST analysis
        validate_code(code)  # Raises SecurityError if unsafe

        # 2. Execute in restricted namespace
        exec(code, self._namespace, self._locals)

        # 3. Capture and return output
        return REPLResult(stdout=..., stderr=...)
```

### 4.4 Message Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Message History                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [System] You are an assistant operating inside a REPL...                │
│                                                                          │
│  [User] Answer: What's the best configuration for my use case?           │
│                                                                          │
│  [Assistant] I'll start by understanding the requirements.               │
│              ```repl                                                     │
│              info = fetch("requirements")                                │
│              print(json.dumps(info, indent=2))                           │
│              ```                                                         │
│                                                                          │
│  [User] Code executed:                                                   │
│         REPL output:                                                     │
│         {                                                                │
│           "min_memory": 16000,                                           │
│           "features": ["A", "B", "C"]                                    │
│         }                                                                │
│                                                                          │
│  [Assistant] I see the requirements. Let me check available options.     │
│              ```repl                                                     │
│              options = search("compatible configurations")               │
│              for opt in options[:5]:                                     │
│                  print(f"{opt['name']}: {opt['memory']}MB")              │
│              ```                                                         │
│                                                                          │
│  [User] Code executed:                                                   │
│         REPL output:                                                     │
│         Option A: 12000MB                                                │
│         Option B: 18000MB                                                │
│         ...                                                              │
│                                                                          │
│  ... (continues until FINAL)                                             │
│                                                                          │
│  [Assistant] FINAL({                                                     │
│      "answer": "Option B is the best fit",                               │
│      "reasoning_steps": [...],                                           │
│      "sources": ["fetch('requirements')", "search(...)"]                 │
│  })                                                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Security: Sandboxing Code Execution

### 5.1 The Risk

RLM executes LLM-generated code. This is inherently risky:

- **Arbitrary code execution** — LLM could generate malicious code
- **Resource exhaustion** — infinite loops, memory bombs
- **Data exfiltration** — accessing files, network requests
- **Sandbox escape** — exploiting Python internals

### 5.2 Defense in Depth

Use multiple layers of protection:

#### Layer 1: Safe Builtins

Only expose a subset of Python builtins:

```python
_SAFE_BUILTINS = {
    # Allowed
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,

    # Explicitly blocked (set to None)
    "input": None,      # No user input
    "eval": None,       # No dynamic eval
    "exec": None,       # No dynamic exec
    "compile": None,    # No code compilation
    "open": None,       # No file access
    "__import__": None, # No imports
    "globals": None,    # No namespace access
    "locals": None,     # No namespace access
}
```

**Subtle security decisions (important lessons):**

```python
# "format" is REMOVED — enables sandbox escape:
# "{0.__class__.__bases__[0].__subclasses__()}".format([])
# This exposes all Python classes!

# "hasattr" is REMOVED — enables attribute probing:
# hasattr(obj, "__class__")  # Can probe for dangerous attributes
```

#### Layer 2: AST Validation

Before execution, analyze the code's Abstract Syntax Tree:

```python
import ast

def validate_code(code: str) -> None:
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SecurityError("Imports are not allowed")

        # Block dangerous attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise SecurityError(f"Dunder access blocked: {node.attr}")

        # Block exec/eval calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('exec', 'eval', 'compile'):
                    raise SecurityError(f"Blocked function: {node.func.id}")
```

#### Layer 3: Function Whitelisting

Only specific tool functions are injected into the namespace:

```python
WHITELISTED_TOOLS = frozenset({
    "search",       # Read-only
    "fetch",        # Read-only
    "status",       # Read-only
})

BLOCKED_TOOLS = frozenset({
    "delete",       # Mutating!
    "update",       # Mutating!
    "execute",      # Mutating!
})
```

If the LLM tries to call a blocked function, it gets a `NameError` because the function simply doesn't exist in the namespace.

#### Layer 4: Execution Timeout

```python
CODE_TIMEOUT_SECONDS = 5

def execute_code(self, code: str) -> REPLResult:
    start = time.perf_counter()

    # ... execute code ...

    elapsed = time.perf_counter() - start
    if elapsed > self._timeout:
        stderr += f"\nTimeoutWarning: execution took {elapsed:.1f}s"
```

Note: Python threads can't be forcibly killed. The timeout is a "soft" limit that warns but doesn't terminate. For true isolation, consider OS-level sandboxing.

#### Layer 5: Output Truncation

Long outputs can overwhelm the context window:

```python
MAX_OUTPUT_CHARS = 10000

output = stdout_buf.getvalue()
if len(output) > MAX_OUTPUT_CHARS:
    output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
```

### 5.3 Advanced: OS-Level Sandboxing

For production systems, consider [sandbox-runtime (srt)](https://github.com/anthropic-experimental/sandbox-runtime):

| Platform | Backend | Description |
|----------|---------|-------------|
| Linux | bubblewrap | User namespace isolation |
| macOS | sandbox-exec | Seatbelt profiles |

```bash
# Run Python in sandbox with no network, limited filesystem
srt --network=deny --write=/tmp python script.py
```

This provides true isolation at the OS level, preventing escapes that Python-level sandboxing might miss.

---

## 6. Building Your Own RLM Application

### 6.1 Step 1: Identify Your Use Case

RLM works best for questions that require:

- **Multi-step reasoning** — Answer depends on intermediate findings
- **Dynamic exploration** — What to look up next depends on what you find
- **Cross-referencing** — Information from multiple sources must be combined
- **Live data** — Current state matters (not just static documents)

**Good RLM use cases:**
- Technical support: "Can X work with Y given Z constraints?"
- Compliance checking: "Does this document meet requirements A, B, C?"
- Debugging: "Why is this system behaving unexpectedly?"
- Configuration: "What's the optimal setup for my situation?"

**Poor RLM use cases:**
- Simple lookups: "What is the capital of France?"
- Single-source answers: Questions answered by one document
- No dependency chains: Answer doesn't require exploration

### 6.2 Step 2: Define Your Tools

List the read-only operations your RLM needs:

```python
# Example tool definitions
def search(query: str) -> list[dict]:
    """Search the knowledge base. Returns list of matches."""
    pass

def fetch(id: str) -> dict:
    """Get detailed information by ID."""
    pass

def status() -> dict:
    """Check current system/resource state."""
    pass

def docs(topic: str) -> str:
    """Fetch documentation on a topic."""
    pass
```

**Design principles:**
- **Read-only** — Never modify state
- **Deterministic** — Same input → same output (when possible)
- **Bounded output** — Limit response size to avoid context overflow
- **Error handling** — Return useful errors, don't crash

### 6.3 Step 3: Write Your System Prompt

Customize the template for your domain:

```python
SYSTEM_PROMPT = f"""
You are a [DOMAIN] assistant operating inside a REPL environment.
Your job is to [PRIMARY TASK] by exploring available data and tools.

## Available Functions

| Function | Description |
|----------|-------------|
{generate_function_table(your_tools)}

Available modules: `json`, `re`, `math`

## Rules

1. **Explore first.** Always call at least one function before answering.
2. **Cite sources.** Reference which function call provided each fact.
3. **No imports.** Do not use `import`, `exec`, `eval`, or `open`.
4. **No mutations.** You only provide recommendations; the user decides.
5. **Be concise.** Keep code blocks short and focused.
6. **Use print().** Print intermediate results so you can reason over them.

## Domain-Specific Guidance

[Add guidance specific to your use case]

## Workflow

1. Examine the `context` variable (query + pre-loaded data).
2. Call tool functions to gather data.
3. Reason over results step-by-step.
4. When ready, provide your final answer as:

FINAL({{
  "answer": "...",
  "reasoning_steps": ["step 1", "step 2"],
  "sources": ["source 1", "source 2"]
}})
"""
```

### 6.4 Step 4: Example Exploration Trajectory

Design an example showing how your RLM should reason:

````
Query: "[Your typical user question]"

Turn 1: Understand the request
─────────────────────────────────
```repl
print(f"Query: {context['query']}")
initial = search(context['query'])
print(f"Found {len(initial)} relevant items")
```

Output: Found 5 relevant items

Turn 2: Gather details
─────────────────────────
```repl
for item in initial[:3]:
    details = fetch(item['id'])
    print(f"{item['name']}: {details['key_property']}")
```

Output:
Item A: value_a
Item B: value_b
Item C: value_c

Turn 3: Check constraints
─────────────────────────
```repl
current = status()
print(f"Available: {current['available']}")
print(f"Required: {details['requirement']}")
```

Output:
Available: 100
Required: 80

Turn 4: Final answer
────────────────────
FINAL({
    "answer": "Yes, Item B is the best choice",
    "reasoning_steps": [
        "Found 5 relevant items matching the query",
        "Item B has the best key_property value",
        "Current availability (100) exceeds requirement (80)"
    ],
    "sources": ["search(query)", "fetch(item_b)", "status()"]
})
````

### 6.5 Step 5: Implement and Test

1. **Build the orchestrator** (see Section 7.1)
2. **Implement tool wrappers** (see Section 7.3)
3. **Add security layers** (see Section 5.2)
4. **Test with adversarial inputs** (see Section 8)

---

## 7. Implementation Patterns

### 7.1 Basic RLM Loop

The minimal implementation:

```python
import re
from typing import Any

def run_rlm(
    query: str,
    tools: dict[str, callable],
    llm_client,
    max_turns: int = 5
) -> dict:
    """Minimal RLM implementation."""

    # Build REPL namespace
    namespace = {
        "context": {"query": query},
        "print": print,
        "len": len,
        "json": __import__("json"),
        "re": __import__("re"),
        "math": __import__("math"),
        **tools,  # Inject tool functions
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}"}
    ]

    for turn in range(max_turns):
        # Get LLM response
        response = llm_client.complete(messages)

        # Check for final answer
        if "FINAL(" in response:
            return parse_final_answer(response)

        # Find and execute code blocks
        code_blocks = re.findall(r"```repl(.*?)```", response, re.DOTALL)

        if code_blocks:
            for code in code_blocks:
                output = execute_sandboxed(code.strip(), namespace)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Output:\n{output}"})
        else:
            # No code - nudge the LLM
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "Use ```repl``` blocks to explore, or provide FINAL(...)"
            })

    # Max turns reached
    return {"error": "Max turns reached", "partial": messages}
```

### 7.2 Structured Result Parsing

Parse FINAL answers robustly:

```python
import json

def parse_final_answer(response: str) -> dict:
    """Parse FINAL(...) with fallbacks."""

    # Extract content between FINAL( and )
    match = re.search(r"FINAL\((.*)\)", response, re.DOTALL)
    if not match:
        return {"answer": response, "raw": True}

    content = match.group(1).strip()

    # Try JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try finding JSON within the content
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: treat as plain text
    return {"answer": content, "raw": True}
```

### 7.3 Tool Wrapper Pattern

Wrap your functions for safe injection:

```python
def create_tool_wrappers(backend) -> dict[str, callable]:
    """Create tool wrappers with consistent interface."""

    def search(query: str, limit: int = 10) -> list[dict]:
        """Search the knowledge base (read-only)."""
        results = backend.search(query, limit=limit)
        # Truncate to prevent context overflow
        return results[:limit]

    def fetch(id: str) -> dict:
        """Fetch details by ID (read-only)."""
        return backend.get_by_id(id)

    def status() -> dict:
        """Get current status (read-only)."""
        return backend.get_status()

    # Only return read-only tools!
    return {
        "search": search,
        "fetch": fetch,
        "status": status,
    }
```

### 7.4 Sub-Query Implementation

Implement recursive `llm_query`:

```python
def create_llm_query(llm_client, depth_limit: int = 2):
    """Create an llm_query function with depth limiting."""

    current_depth = 0

    def llm_query(prompt: str, context: str = "") -> str:
        nonlocal current_depth

        if current_depth >= depth_limit:
            return "Error: Maximum recursion depth reached"

        current_depth += 1
        try:
            full_prompt = f"Sub-task: {prompt}"
            if context:
                full_prompt += f"\n\nContext:\n{context[:2000]}"  # Truncate

            response = llm_client.complete([
                {"role": "user", "content": full_prompt}
            ])
            return response
        finally:
            current_depth -= 1

    return llm_query
```

### 7.5 Sandboxed Execution

Execute code safely:

```python
import io
import sys

def execute_sandboxed(code: str, namespace: dict, timeout: float = 5.0) -> str:
    """Execute code with output capture and safety checks."""

    # 1. AST validation
    validate_code(code)  # Raises if unsafe

    # 2. Capture stdout/stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        # 3. Execute with restricted builtins
        safe_namespace = {
            "__builtins__": SAFE_BUILTINS,
            **namespace
        }
        exec(code, safe_namespace, safe_namespace)

        output = stdout_buf.getvalue()
        errors = stderr_buf.getvalue()

        # 4. Truncate if too long
        if len(output) > 10000:
            output = output[:10000] + "\n... (truncated)"

        return output + errors

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
```

---

## 8. Common Pitfalls and Solutions

### 8.1 Output Overflow

**Problem:** LLM prints entire datasets, overwhelming context window.

**Solution:**
```python
# Truncate in execution
MAX_OUTPUT = 5000
output = stdout_buf.getvalue()
if len(output) > MAX_OUTPUT:
    output = output[:MAX_OUTPUT] + f"\n... (truncated, {len(output)} total chars)"
```

Also add guidance in the system prompt:
```
- Never print entire results. Use slicing: results[:10]
- Limit loop iterations: for item in items[:10]:
- Use len() to check size before printing
```

### 8.2 Infinite Loops

**Problem:** LLM generates `while True:` or similar.

**Solution:**
- AST validation to detect unbounded loops
- Execution timeout
- OS-level sandboxing for hard limit

```python
# AST check for while True
for node in ast.walk(tree):
    if isinstance(node, ast.While):
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            raise SecurityError("Unbounded while loop detected")
```

### 8.3 LLM Doesn't Write Code

**Problem:** LLM answers directly instead of exploring.

**Solution:** Strengthen system prompt:
```
## CRITICAL RULES
1. You MUST write at least one ```repl``` code block before answering.
2. NEVER answer based on assumptions. Always verify with code.
3. If you're tempted to answer without code, STOP and write code first.
```

And nudge in the loop:
```python
if not code_blocks:
    messages.append({
        "role": "user",
        "content": "You must use ```repl``` blocks to explore. Please write code."
    })
```

### 8.4 Stuck in a Loop

**Problem:** LLM keeps exploring without reaching FINAL.

**Solution:**
- Hard limit on turns (`max_turns`)
- Force final answer when limit reached:

```python
if turn == max_turns - 1:
    messages.append({
        "role": "user",
        "content": "IMPORTANT: This is your last turn. "
                   "Provide your FINAL(...) answer now based on what you've gathered."
    })
```

### 8.5 Import Attempts

**Problem:** LLM tries to `import os` or similar.

**Solution:**
- AST validation blocks all imports
- `__import__` set to `None` in builtins
- Clear system prompt guidance

```python
if isinstance(node, (ast.Import, ast.ImportFrom)):
    raise SecurityError(f"Imports blocked: {ast.dump(node)}")
```

### 8.6 Sandbox Escape Attempts

**Problem:** LLM tries `"".__class__.__bases__[0].__subclasses__()` to access dangerous classes.

**Solution:**
- Block dunder attribute access in AST
- Remove `format` from builtins (template injection)
- Remove `hasattr` from builtins (attribute probing)

```python
if isinstance(node, ast.Attribute):
    if node.attr.startswith('_'):
        raise SecurityError(f"Private/dunder access blocked: {node.attr}")
```

---

## 9. References

### Papers and Repositories

- **RLM Paper**: Zhang et al. (2025) - [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- **sandbox-runtime**: Anthropic - [github.com/anthropic-experimental/sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)

### Related Concepts

- **ReAct Pattern**: Reasoning + Acting — LLMs that interleave reasoning with tool use
- **Code Interpreters**: Systems like ChatGPT Code Interpreter that execute generated code
- **Agentic AI**: AI systems that take actions in environments

---

## Appendix A: Quick Start Checklist

When implementing RLM for a new project:

- [ ] **Define use case** — What questions require multi-step reasoning?
- [ ] **Design tools** — What read-only operations does the LLM need?
- [ ] **Write system prompt** with:
  - [ ] Available functions table
  - [ ] Domain-specific guidance
  - [ ] Rules (no imports, use print, etc.)
  - [ ] FINAL answer format
- [ ] **Implement safe builtins** (start restrictive, add as needed)
- [ ] **Add AST validation** for imports, dunders, dangerous calls
- [ ] **Create tool wrappers** (read-only only!)
- [ ] **Implement the REPL loop** with:
  - [ ] Code block extraction
  - [ ] Sandboxed execution
  - [ ] Output capture and feedback
  - [ ] FINAL detection
- [ ] **Add safeguards**:
  - [ ] Max turns limit
  - [ ] Output truncation
  - [ ] Execution timeout
- [ ] **Test adversarial inputs** (import attempts, loops, escapes)

---

## Appendix B: System Prompt Template

Copy and customize for your project:

```markdown
You are a [DOMAIN] assistant operating inside a REPL environment.
Your task is to answer questions by exploring available tools and data.

## Available Functions

| Function | Description |
|----------|-------------|
| `context` | The query and any pre-loaded data |
| `tool_a(param)` | Description of tool A |
| `tool_b(param)` | Description of tool B |
| `llm_query(prompt, data)` | Ask a focused sub-question |

Available modules: `json`, `re`, `math`

## Rules

1. **Explore first.** Always write code before answering.
2. **Use print().** Results must be printed to be visible.
3. **No imports.** Do not use `import`, `exec`, `eval`, or `open`.
4. **Be concise.** Keep code blocks short and focused.
5. **Cite sources.** Reference which tool calls provided your data.

## Workflow

1. Examine `context` to understand the query.
2. Call tool functions to gather data.
3. Use `llm_query()` for detailed sub-analysis if needed.
4. When ready, output your answer as:

FINAL({
    "answer": "...",
    "reasoning": ["step 1", "step 2"],
    "sources": ["tool_a()", "tool_b()"]
})

Begin exploring now.
```

---

## Appendix C: Complete Minimal Implementation

A complete, copy-paste ready implementation:

```python
"""
Minimal RLM Implementation
Copy this file to start your own RLM project.
"""

import ast
import io
import json
import re
import sys
import time
from typing import Any, Callable

# =============================================================================
# SAFE BUILTINS
# =============================================================================

SAFE_BUILTINS: dict[str, Any] = {
    # Core types
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "type": type,

    # Iteration
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "iter": iter,
    "next": next,

    # Sorting/ordering
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,

    # Math
    "sum": sum,
    "abs": abs,
    "round": round,

    # Logic
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "callable": callable,

    # String
    "repr": repr,

    # Exceptions (for error handling in generated code)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,

    # BLOCKED - explicitly set to None
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "open": None,
    "__import__": None,
    "globals": None,
    "locals": None,
    # NOTE: "format" and "hasattr" intentionally omitted (security risk)
}


# =============================================================================
# AST VALIDATION
# =============================================================================

class SecurityError(Exception):
    """Raised when code fails security validation."""
    pass


def validate_code(code: str) -> None:
    """Validate code via AST analysis. Raises SecurityError if unsafe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"Syntax error: {e}")

    for node in ast.walk(tree):
        # Block imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SecurityError("Imports are not allowed")

        # Block dunder attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise SecurityError(f"Private/dunder access blocked: {node.attr}")

        # Block dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ('exec', 'eval', 'compile', 'open', '__import__'):
                    raise SecurityError(f"Blocked function: {node.func.id}")


# =============================================================================
# SANDBOXED EXECUTION
# =============================================================================

def execute_sandboxed(
    code: str,
    namespace: dict[str, Any],
    max_output: int = 10000,
    timeout: float = 5.0,
) -> str:
    """Execute code in sandbox with output capture."""

    # 1. Validate
    validate_code(code)

    # 2. Setup output capture
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    start = time.perf_counter()

    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        # 3. Execute with safe builtins
        safe_ns = {"__builtins__": SAFE_BUILTINS, **namespace}
        exec(code, safe_ns, safe_ns)

        # 4. Update namespace with new variables
        for key, value in safe_ns.items():
            if key not in namespace and not key.startswith('_'):
                namespace[key] = value

        output = stdout_buf.getvalue()
        errors = stderr_buf.getvalue()

    except Exception as e:
        output = stdout_buf.getvalue()
        errors = f"{type(e).__name__}: {e}"

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # 5. Truncate if needed
    result = output + errors
    if len(result) > max_output:
        result = result[:max_output] + f"\n... (truncated, {len(result)} total)"

    # 6. Timeout warning
    elapsed = time.perf_counter() - start
    if elapsed > timeout:
        result += f"\nTimeoutWarning: took {elapsed:.1f}s"

    return result


# =============================================================================
# RESULT PARSING
# =============================================================================

def parse_final_answer(response: str) -> dict[str, Any]:
    """Parse FINAL(...) from response with fallbacks."""

    match = re.search(r"FINAL\((.*)\)", response, re.DOTALL)
    if not match:
        return {"answer": response, "raw": True}

    content = match.group(1).strip()

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from content
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"answer": content, "raw": True}


# =============================================================================
# MAIN RLM LOOP
# =============================================================================

def run_rlm(
    query: str,
    tools: dict[str, Callable],
    llm_complete: Callable[[list[dict]], str],
    system_prompt: str,
    max_turns: int = 5,
) -> dict[str, Any]:
    """
    Run the RLM loop.

    Args:
        query: User's question
        tools: Dict of tool_name -> callable (read-only functions)
        llm_complete: Function that takes messages list, returns response string
        system_prompt: System prompt teaching the LLM how to use the REPL
        max_turns: Maximum iterations before forcing final answer

    Returns:
        Parsed FINAL answer as dict, or error dict
    """

    # Build namespace
    namespace: dict[str, Any] = {
        "context": {"query": query},
        "json": json,
        "re": re,
        **tools,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}"},
    ]

    for turn in range(max_turns):
        # Get LLM response
        response = llm_complete(messages)

        # Check for final answer
        if "FINAL(" in response:
            result = parse_final_answer(response)
            result["turns_used"] = turn + 1
            return result

        # Extract code blocks
        code_blocks = re.findall(r"```repl(.*?)```", response, re.DOTALL)

        messages.append({"role": "assistant", "content": response})

        if code_blocks:
            for code in code_blocks:
                output = execute_sandboxed(code.strip(), namespace)
                messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code.strip()}\n```\n\nOutput:\n{output}"
                })
        else:
            # Nudge to write code
            messages.append({
                "role": "user",
                "content": "Use ```repl``` code blocks to explore, or provide FINAL(...)"
            })

        # Force final on last turn
        if turn == max_turns - 1:
            messages.append({
                "role": "user",
                "content": "IMPORTANT: This is your last turn. Provide FINAL(...) now."
            })

    # Exhausted turns - try one more time for final
    response = llm_complete(messages)
    if "FINAL(" in response:
        result = parse_final_answer(response)
        result["turns_used"] = max_turns + 1
        return result

    return {
        "error": "Max turns reached without FINAL answer",
        "turns_used": max_turns + 1,
        "last_response": response,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Define your tools
    def search(query: str, limit: int = 10) -> list[dict]:
        """Search your knowledge base."""
        # Replace with your actual search implementation
        return [{"id": "1", "title": "Example", "score": 0.95}]

    def fetch(id: str) -> dict:
        """Fetch details by ID."""
        # Replace with your actual fetch implementation
        return {"id": id, "content": "Example content", "metadata": {}}

    def status() -> dict:
        """Get current system status."""
        # Replace with your actual status check
        return {"available": True, "resources": {"memory": 1000}}

    tools = {
        "search": search,
        "fetch": fetch,
        "status": status,
    }

    # Example: Define your system prompt
    SYSTEM_PROMPT = """
You are an assistant operating inside a REPL environment.

## Available Functions

| Function | Description |
|----------|-------------|
| `search(query, limit=10)` | Search the knowledge base |
| `fetch(id)` | Get details by ID |
| `status()` | Check system status |

Available modules: `json`, `re`

## Rules

1. **Explore first.** Call functions before answering.
2. **Use print().** Results must be printed to be visible.
3. **No imports.** Don't use import, exec, eval, open.
4. **Cite sources.** Reference which function calls provided data.

## Workflow

1. Examine `context['query']`
2. Call tool functions
3. Output FINAL({...}) when ready

FINAL({
    "answer": "...",
    "reasoning": ["step 1", "step 2"],
    "sources": ["search(...)", "fetch(...)"]
})
"""

    # Example: Mock LLM for testing
    def mock_llm_complete(messages: list[dict]) -> str:
        # Replace with actual LLM API call
        # e.g., openai.chat.completions.create(...)
        return """
I'll search for relevant information.

```repl
results = search(context['query'])
print(f"Found {len(results)} results")
for r in results:
    print(f"- {r['title']} (score: {r['score']})")
```
"""

    # Run it
    result = run_rlm(
        query="What information do you have?",
        tools=tools,
        llm_complete=mock_llm_complete,
        system_prompt=SYSTEM_PROMPT,
        max_turns=5,
    )

    print(json.dumps(result, indent=2))
```

---

*This guide provides everything needed to implement RLM in any project.*
