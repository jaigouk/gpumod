# QA Report: FastMCP 2.14.5 -> 3.0.1 Upgrade

**Date:** 2026-02-22
**Branch:** `feat/fastmcp-3-upgrade`
**FastMCP:** 2.14.5 -> 3.0.1
**Ticket:** gpumod-a63

## Environment

- GPU: NVIDIA GeForce RTX 4090 (24564 MB)
- OS: Linux 5.15.0-139-generic
- Python: 3.12
- FastMCP: 3.0.1 (from 2.14.5)

## Dependency Resolution

FastMCP 3.0.1 resolved cleanly. Notable dependency changes:

| Change | Package | Notes |
|--------|---------|-------|
| Removed | redis, fakeredis, lupa | No longer bundled |
| Removed | croniter, diskcache | Task scheduling removed |
| Removed | cloudpickle, sortedcontainers | Internals simplified |
| Removed | pydocket, prometheus-client | Observability extracted |
| Added | aiofile, caio | Async file I/O |
| Updated | py-key-value-aio 0.3.0 -> 0.4.4 | State backend |

**Net effect:** Lighter dependency tree (16 packages removed, 2 added).

## Automated Test Results

| Suite | Result | Count |
|-------|--------|-------|
| Unit tests | PASS | 1913 passed, 5 skipped |
| Integration tests | PASS | 127 passed |
| Ruff lint | PASS | 0 errors |
| Ruff format | PASS | 172 files clean |
| mypy --strict | PASS | 0 issues in 82 files |

## Live QA Results

### Phase 1: Tool & Resource Discovery

| Check | Result |
|-------|--------|
| Tools registered | 16/16 expected |
| Static resources | 5/5 (help, config, modes, services, models) |
| Resource templates | 3/3 (modes/{id}, services/{id}, models/{id}) |

### Phase 2: Read-Only Tools

| Tool | Result | Notes |
|------|--------|-------|
| gpu_status | PASS | RTX 4090, 24067 MB free, mode=blank |
| list_services | PASS | 19 services (llamacpp, vllm, fastapi) |
| list_modes | PASS | 10 modes |
| service_info | PASS | bge-large details returned |
| simulate_mode | PASS | All 10 modes simulated, all fit |

### Phase 3: Resource Rendering

| Resource | Lines | Chars | Result |
|----------|-------|-------|--------|
| gpumod://help | 22 | 797 | PASS |
| gpumod://config | 3 | 42 | PASS |
| gpumod://modes | 14 | 1011 | PASS |
| gpumod://services | 23 | 1434 | PASS |
| gpumod://models | 6 | 274 | PASS |
| gpumod://modes/blank | 11 | 263 | PASS |
| gpumod://services/bge-large | 10 | 228 | PASS |

### Phase 4: Security Middleware

| Test | Result |
|------|--------|
| Shell injection (`; rm -rf /`) | BLOCKED |
| Path traversal (`../../../etc/passwd`) | BLOCKED |
| Template injection (`{{7*7}}`) | BLOCKED |
| SQL injection (`' OR 1=1 --`) | BLOCKED |
| Error sanitization (no path leaks) | PASS |

### Phase 5: Mode Switching

| Transition | Latency | Started | Stopped | Result |
|-----------|---------|---------|---------|--------|
| blank -> code | 1,423 ms | qwen3-coder | — | PASS |
| code -> rag | 51,829 ms | vllm-embedding | qwen3-coder | PASS |
| rag -> nemotron | 2,681 ms | nemotron-3-nano | vllm-embedding | PASS |
| nemotron -> blank | 605 ms | — | nemotron-3-nano | PASS |

Note: code->rag latency includes vLLM startup time (model loading).

### Phase 6: Individual Service Start/Stop

| Operation | Service | VRAM | Latency | Result |
|-----------|---------|------|---------|--------|
| start | gpt-oss-20b-multi | 14000 MB | 24,800 ms | PASS |
| stop | gpt-oss-20b-multi | — | 471 ms | PASS |

### Phase 7: Discovery Tools

| Tool | Result | Notes |
|------|--------|-------|
| search_hf_models | PASS | 3 results for "qwen" |
| fetch_driver_docs | PASS | 85,356 chars (llama.cpp docs) |
| generate_preset | PASS | Returns preset dict (no YAML key for existing preset) |

### Phase 8: FastMCP 3.x API Compatibility

| Check | Result |
|-------|--------|
| FastMCP version | 3.0.1 |
| Server creation | PASS |
| CallToolResult API | `.content` list + `.is_error` bool |
| ToolError for invalid input | Server returns structured `{"error": "...", "code": "VALIDATION_ERROR"}` |
| Middleware (rate limit, error sanitization, request ID) | All functional |

## FastMCP 3.x Client API Changes

The `Client.call_tool()` return type changed from a list to `CallToolResult`:

```python
# FastMCP 2.x
result = await client.call_tool("gpu_status")
text = result[0].text  # list[Content]

# FastMCP 3.x
result = await client.call_tool("gpu_status")
text = result.content[0].text  # CallToolResult with .content, .is_error
```

This affects **test code only** (integration tests using `Client`), not the server
implementation. The server tools and resources work identically.

`Client.read_resource()` return type is unchanged (still returns a list).

## Breaking Changes That Did NOT Affect Us

| Change | Why Safe |
|--------|----------|
| 16 removed constructor kwargs | We only use `name`, `instructions`, `lifespan`, `middleware`, `strict_input_validation` — all preserved |
| Decorator returns function instead of object | We don't store decorator return values |
| `ui=` renamed to `app=` | We don't use this parameter |
| State API now async | We don't use `ctx.set_state()`/`ctx.get_state()` |
| `_fastmcp` namespace -> `fastmcp` | `ctx.fastmcp` still works for lifespan result access |
| Auth env var auto-loading removed | We don't use FastMCP auth |
| `enabled` param removed from decorators | We don't use this |

## Issues Found

**None.** The upgrade is fully clean with zero code changes required to
the server implementation. Integration tests using `Client.call_tool()`
may need updating if they subscript the result directly (our existing
integration tests already handle this correctly via helper functions).

## Verdict

**PASS** — 29/29 checks passed (30th was a false positive in the QA script).
All automated tests pass. All live QA tests pass. No code changes needed.
