#!/usr/bin/env python3
"""Test multi-agent Tetris development with 3 parallel slots.

Agent 1: Planner - designs the architecture
Agent 2: Developer - implements the code
Agent 3: QA - reviews and tests
"""
from __future__ import annotations

import asyncio
import json
import httpx

BASE_URL = "http://localhost:7070/v1/chat/completions"


async def call_llm(role: str, messages: list[dict], max_tokens: int = 2000) -> str:
    """Call the LLM with given messages."""
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            BASE_URL,
            json={
                "model": "qwen3-coder",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"\n{'='*60}")
        print(f"[{role}] Response ({len(content)} chars)")
        print(f"{'='*60}")
        return content


async def planner() -> str:
    """Agent 1: Plan the Tetris architecture."""
    messages = [
        {
            "role": "user",
            "content": """Design a terminal-based Tetris game in Python.

Requirements:
- Pure Python, no external dependencies (use curses for terminal)
- Standard Tetris pieces (I, O, T, S, Z, J, L)
- Arrow keys for movement, up for rotation
- Line clearing and scoring
- Game over detection

Provide a detailed architecture plan with:
1. Class structure
2. Key functions
3. Game loop design
4. Data structures for pieces and board

Be concise but complete. Output ONLY the plan, no code yet."""
        }
    ]
    return await call_llm("PLANNER", messages, max_tokens=1500)


async def developer(plan: str) -> str:
    """Agent 2: Implement the Tetris game based on plan."""
    messages = [
        {
            "role": "user",
            "content": f"""Based on this architecture plan, implement a complete terminal Tetris game:

{plan}

Requirements:
- Single Python file
- Use curses for terminal rendering
- Must be fully functional and runnable
- Include all standard Tetris pieces
- Handle keyboard input for movement/rotation

Output ONLY the complete Python code, no explanations. Start with #!/usr/bin/env python3"""
        }
    ]
    return await call_llm("DEVELOPER", messages, max_tokens=4000)


async def qa_review(code: str) -> str:
    """Agent 3: Review the code for issues."""
    messages = [
        {
            "role": "user",
            "content": f"""Review this Tetris game code for bugs and issues:

```python
{code}
```

Check for:
1. Syntax errors
2. Logic bugs (collision detection, line clearing, rotation)
3. Missing imports
4. Runtime errors
5. Playability issues

List any critical issues found. If the code looks correct, say "LGTM - code appears functional"."""
        }
    ]
    return await call_llm("QA", messages, max_tokens=1000)


async def main():
    print("Starting multi-agent Tetris development...")
    print("Using 3 parallel slots on Qwen3-Coder-30B")

    # Phase 1: Planning
    print("\n" + "="*60)
    print("PHASE 1: PLANNING")
    print("="*60)
    plan = await planner()
    print(plan[:2000] + "..." if len(plan) > 2000 else plan)

    # Phase 2: Development (run in parallel with QA prep)
    print("\n" + "="*60)
    print("PHASE 2: DEVELOPMENT")
    print("="*60)
    code = await developer(plan)

    # Extract code block if wrapped
    if "```python" in code:
        start = code.find("```python") + 9
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()
    elif "```" in code:
        start = code.find("```") + 3
        end = code.find("```", start)
        if end > start:
            code = code[start:end].strip()

    # Save the code
    with open("tetris_game.py", "w") as f:
        f.write(code)
    print(f"Code saved to tetris_game.py ({len(code)} chars)")
    print(code[:1500] + "..." if len(code) > 1500 else code)

    # Phase 3: QA Review
    print("\n" + "="*60)
    print("PHASE 3: QA REVIEW")
    print("="*60)
    review = await qa_review(code)
    print(review)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Plan: {len(plan)} chars")
    print(f"Code: {len(code)} chars")
    print(f"Review: {len(review)} chars")
    print(f"\nTetris game saved to: tetris_game.py")
    print("To test: python tetris_game.py")


if __name__ == "__main__":
    asyncio.run(main())
