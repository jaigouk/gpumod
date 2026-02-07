# gpumod Security Model for MCP Tools

This document defines the threat model, input validation specification, and security
requirements for exposing gpumod operations via MCP (Model Context Protocol) tools.

All Phase 4 implementation tickets (T1-T6) **must** reference this document and follow
its specifications.

---

## 1. Threat Model

MCP tools are invoked by LLMs, which may relay untrusted user input or be subject to
prompt injection. Every tool argument, resource URI parameter, and return value is an
attack surface.

| # | Threat | Vector | Impact | Mitigation | Ref |
|---|--------|--------|--------|------------|-----|
| T1 | Shell injection via service/mode IDs | LLM passes `"; rm -rf /"` as `service_id` to a tool that eventually reaches `systemctl` | Arbitrary command execution | SEC-V1: Strict regex validation on all ID args at MCP boundary; systemd.py allowlist + unit name regex as defense-in-depth | SEC-V1, SEC-D1 |
| T2 | SQL injection via string args | LLM passes `"'; DROP TABLE services--"` as tool arg | Data loss, DB corruption | SEC-V1: Regex rejects SQL metacharacters; SEC-D2: aiosqlite parameterized queries as defense-in-depth | SEC-V1, SEC-D2 |
| T3 | Path traversal via model_id | LLM passes `"../../etc/passwd"` as `model_id` | File system read/write | SEC-V1: model_id regex allows only `[a-zA-Z0-9_\-./]`; no file path construction from model_id in MCP layer | SEC-V1 |
| T4 | Jinja2 template injection | LLM passes `"{{7*7}}"` as a string arg | Code execution in sandboxed environment | SEC-V1: Regex rejects `{` and `}` in IDs; SEC-D3: SandboxedEnvironment as defense-in-depth | SEC-V1, SEC-D3 |
| T5 | Information disclosure via errors | Internal exception bubbles up with DB path, traceback | Leaks internal architecture | SEC-E1: Error sanitization middleware strips paths and tracebacks | SEC-E1 |
| T6 | Information disclosure via resources | Resource output includes file paths or internal config | Leaks deployment details | SEC-E2: Resource output must not contain absolute paths | SEC-E2 |
| T7 | Unauthorized mutating operations | LLM calls `switch_mode` or `stop_service` without user intent | Service disruption | SEC-A1: Tool classification (read-only vs mutating); LLM client must confirm mutating actions | SEC-A1 |
| T8 | Resource exhaustion via simulation | LLM calls `simulate_mode` in a loop, generating max alternatives each time | CPU/memory spike | SEC-R1: Cap max alternatives at 10; SEC-R2: Rate limit middleware | SEC-R1, SEC-R2 |
| T9 | Resource exhaustion via MCP flooding | LLM sends rapid-fire tool calls | Server overload | SEC-R2: Rate limit middleware (default 10 req/s) | SEC-R2 |
| T10 | Terminal escape injection | Service/mode names with ANSI escapes returned to LLM, then displayed to user | Terminal manipulation | SEC-E3: Sanitize names in all output (existing `_sanitize_name()` pattern) | SEC-E3 |
| T11 | Extra fields in tool input | LLM sends unexpected kwargs that bypass validation | Logic bypass | SEC-V2: FastMCP `strict_input_validation=True`; Pydantic `extra="forbid"` on all models | SEC-V2 |

---

## 2. Input Validation Specification (SEC-V1)

Every string argument to an MCP tool or resource template **must** be validated against
the corresponding regex at the MCP boundary, **before** any business logic executes.

### 2.1 ID Validation Regexes

| Argument | Regex | Max Length | Examples (valid) | Examples (rejected) |
|----------|-------|------------|------------------|---------------------|
| `service_id` | `^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$` | 64 | `vllm-chat-01`, `fastapi_app` | `../etc`, `'; DROP`, `{{7*7}}` |
| `mode_id` | `^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$` | 64 | `chat-mode`, `code_dev` | `; rm -rf /`, empty string |
| `model_id` | `^[a-zA-Z0-9][a-zA-Z0-9_\-./]{0,127}$` | 128 | `meta-llama/Llama-3.1-8B`, `my-gguf-q4` | `../../passwd`, `$(cmd)` |

### 2.2 Numeric Validation

| Argument | Type | Min | Max | Notes |
|----------|------|-----|-----|-------|
| `context_overrides` values | `int` | 1 | 131072 | Context window size in tokens |
| `context_overrides` keys | `str` | — | — | Validated as `service_id` |

### 2.3 Validation Helper

All MCP modules should use a shared validation module to avoid duplication:

```python
# src/gpumod/validation.py
import re

SERVICE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
MODE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-./]{0,127}$")
MAX_CONTEXT_TOKENS = 131072

def validate_service_id(value: str) -> str: ...
def validate_mode_id(value: str) -> str: ...
def validate_model_id(value: str) -> str: ...
def validate_context_override(key: str, value: int) -> tuple[str, int]: ...
```

### 2.4 Strict Input Mode (SEC-V2)

- FastMCP server **must** be created with `strict_input_validation=True`
- All Pydantic models **must** use `ConfigDict(extra="forbid")` (already enforced)

---

## 3. Tool Classification (SEC-A1)

### 3.1 Read-Only Tools (always safe)

These tools perform no mutations and can be called freely:

| Tool | Description | Risk Level |
|------|-------------|------------|
| `gpu_status` | Current GPU status, VRAM, services | None |
| `list_services` | All registered services | None |
| `list_modes` | All available modes | None |
| `service_info` | Single service detail | None |
| `model_info` | Single model detail | None |
| `simulate_mode` | VRAM simulation (non-destructive) | Low (CPU) |

### 3.2 Mutating Tools (require confirmation)

These tools change system state. LLM clients (Claude Desktop, etc.) should present
confirmation UIs before executing. The MCP server marks these in tool descriptions.

| Tool | Description | Risk Level | Side Effects |
|------|-------------|------------|--------------|
| `switch_mode` | Start/stop services | High | Starts/stops systemd units |
| `start_service` | Start a service | Medium | Starts a systemd unit |
| `stop_service` | Stop a service | Medium | Stops a systemd unit |

### 3.3 Audit Logging

All mutating tool calls **must** be logged at `INFO` level with:
- Tool name
- Arguments (after validation)
- Caller context (if available from MCP)
- Result (success/failure)

---

## 4. Output Sanitization

### 4.1 Error Responses (SEC-E1)

Error messages returned to the LLM **must not** contain:
- Absolute file paths (DB path, model file path, template path)
- Python tracebacks or stack frames
- Internal module names or line numbers

Pattern:
```python
# Bad:  "Database error: /home/user/.config/gpumod/gpumod.db: table not found"
# Good: "Database error: operation failed"

# Bad:  "Traceback (most recent call last):\n  File '/home/...'"
# Good: "Internal error: please check server logs"
```

Implementation: `ErrorSanitizationMiddleware` strips paths matching common patterns
(`/home/`, `/tmp/`, `/var/`, `.db`, `.py`) and replaces with generic messages.

### 4.2 Resource Output (SEC-E2)

MCP resource content **must not** contain:
- Absolute file paths to DB, config, or model files
- Environment variable values
- Credentials or tokens

Resources may contain:
- Service/mode/model IDs
- VRAM numbers
- Configuration parameters (port, context_size, etc.)

### 4.3 Name Sanitization (SEC-E3)

All service, mode, and model names in MCP output **must** be sanitized:
- Strip ANSI escape sequences
- Strip Rich markup tags
- Remove control characters

Use the existing `visualization._sanitize_name()` pattern, extracted to `validation.py`.

---

## 5. Rate Limiting (SEC-R1, SEC-R2)

### 5.1 Simulation Alternatives Cap (SEC-R1)

`SimulationEngine._generate_alternatives()` **must** cap output at **10 alternatives**.
This prevents CPU exhaustion from combinatorial exploration.

### 5.2 Request Rate Limit (SEC-R2)

The MCP server **should** implement rate limiting middleware:
- Default: 10 requests per second per connection
- Configurable via environment variable `GPUMOD_MCP_RATE_LIMIT`
- Returns clear error when limit exceeded: `{"error": "Rate limit exceeded", "code": "RATE_LIMITED"}`

---

## 6. Existing Security Controls Audit

### 6.1 Strengths (defense-in-depth)

| Control | Location | Status | Notes |
|---------|----------|--------|-------|
| **SEC-D1**: systemctl command allowlist | `services/systemd.py:14-25` | Good | 8 commands whitelisted, all others rejected |
| **SEC-D1**: Unit name regex | `services/systemd.py:27` | Good | `^[a-zA-Z0-9_@:.\-]+$` rejects shell metacharacters |
| **SEC-D1**: No `shell=True` | `services/systemd.py:95` | Good | Uses `create_subprocess_exec` throughout |
| **SEC-D2**: Parameterized SQL queries | `db.py` (all methods) | Good | All queries use `?` placeholders, no string interpolation |
| **SEC-D3**: Jinja2 SandboxedEnvironment | `templates/engine.py:44` | Good | Prevents template code execution |
| **SEC-D3**: Template name validation | `templates/engine.py:57-72` | Good | Rejects path traversal and absolute paths |
| **SEC-D4**: Pydantic `extra="forbid"` | `models.py` (all models) | Good | Rejects unexpected fields |
| **SEC-D5**: Name sanitization | `visualization.py:38-59` | Good | Strips ANSI, Rich markup, control chars |

### 6.2 Gaps (addressed by this spec)

| Gap | Severity | Addressed By |
|-----|----------|--------------|
| No input validation at business logic boundary | Medium | SEC-V1: MCP tools validate all args; `validation.py` module |
| Error messages may leak internal paths | Medium | SEC-E1: ErrorSanitizationMiddleware |
| No rate limiting | Low | SEC-R2: RateLimitMiddleware |
| No audit logging for mutations | Low | SEC-A1: Audit log for mutating tools |
| `_sanitize_name` not reusable outside visualization | Low | SEC-E3: Extract to `validation.py` |

---

## 7. Recommended Deployment Configuration

### 7.1 Transport

| Mode | Use Case | Security |
|------|----------|----------|
| **stdio** (default) | Claude Desktop, local LLM clients | Process-level isolation, no network exposure |
| **SSE** | Remote or multi-client | **Must** bind to `localhost` only; use reverse proxy + TLS for remote access |

### 7.2 MCP Client Configuration

```json
{
  "mcpServers": {
    "gpumod": {
      "command": "python",
      "args": ["-m", "gpumod.mcp_main"],
      "env": {
        "GPUMOD_DB_PATH": "~/.config/gpumod/gpumod.db",
        "GPUMOD_MCP_RATE_LIMIT": "10"
      }
    }
  }
}
```

---

## 8. Implementation Checklist

Tickets **must** check off relevant items before closing.

### Input Validation
- [ ] `validation.py` module created with shared regexes and validators
- [ ] All MCP tool string args validated via SEC-V1 before business logic
- [ ] All MCP resource template params validated via SEC-V1
- [ ] `SimulationEngine` validates IDs via SEC-V1 before DB lookups
- [ ] CLI `simulate` command validates `--add`/`--remove` IDs via SEC-V1
- [ ] FastMCP created with `strict_input_validation=True` (SEC-V2)

### Output Sanitization
- [ ] `ErrorSanitizationMiddleware` strips internal paths (SEC-E1)
- [ ] MCP resources contain no absolute paths (SEC-E2)
- [ ] Service/mode/model names sanitized in all MCP output (SEC-E3)

### Authorization & Audit
- [ ] Tools classified as read-only or mutating in descriptions (SEC-A1)
- [ ] Mutating tools log operations at INFO level (SEC-A1)

### Rate Limiting & Resource Protection
- [ ] Simulation alternatives capped at 10 (SEC-R1)
- [ ] Rate limit middleware implemented (SEC-R2)

### Testing
- [ ] Input validation tests: shell injection, SQL injection, template injection, path traversal
- [ ] Error sanitization tests: no internal paths in error responses
- [ ] Rate limit tests: excess requests rejected
- [ ] Integration tests verify end-to-end security controls
