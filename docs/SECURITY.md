# gpumod Security Model

This document defines the threat model, input validation specification, and security
requirements for gpumod. Covers MCP tool exposure (Phase 4) and LLM-facing AI
planning (Phase 5).

All implementation tickets **must** reference this document and follow its specifications.

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

### 1.2 LLM Integration Threats (Phase 5)

The `gpumod plan` command calls external LLM APIs (OpenAI, Anthropic, Ollama) and uses
their responses to generate VRAM allocation plans. This creates additional attack surfaces.

| # | Threat | Vector | Impact | Mitigation | Ref |
|---|--------|--------|--------|------------|-----|
| T12 | Indirect prompt injection via DB-stored names | Attacker stores malicious service/mode name in DB (e.g., `"ignore previous instructions..."`) that gets included in LLM prompt | LLM produces manipulated plan output | SEC-L2: Prompt template hardening with clear instruction boundaries; SEC-L1: validate all IDs in response regardless of LLM output | SEC-L1, SEC-L2 |
| T13 | LLM response manipulation | LLM returns malicious plan (e.g., stop all services, allocate impossible VRAM) | Service disruption if auto-executed | SEC-L4: LLM output is advisory only, never auto-executed; SimulationEngine validates plan feasibility | SEC-L4 |
| T14 | API key leakage | API key appears in error messages, logs, or LLM prompt context | Credential theft, unauthorized API usage | SEC-L3: API key stored as SecretStr, never logged, never in exception messages | SEC-L3 |
| T15 | LLM response parsing injection | LLM returns crafted JSON with unexpected fields or injection payloads in ID fields | ID validation bypass, potential shell injection downstream | SEC-L1: All IDs in LLM response validated via validation.py before use; Pydantic strict parsing | SEC-L1 |
| T16 | Excessive LLM API calls (cost exhaustion) | Automated or rapid-fire `gpumod plan suggest` calls | Unexpected API billing costs | SEC-R2: Rate limiting; CLI is interactive (human-speed); MCP rate limit middleware | SEC-R2 |
| T17 | Sensitive data sent to external LLM APIs | Full DB dump or internal config sent as prompt context | Data exfiltration to third-party API | SEC-L5: Data minimization — only service IDs, VRAM numbers, and mode names sent; no paths, credentials, or full configs | SEC-L5 |

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

## 5. LLM Security Controls (SEC-L1 through SEC-L5)

### 5.1 LLM Response Validation (SEC-L1)

All IDs in LLM-generated plans **must** be validated via `validation.py` before use:
- Service IDs validated against `SERVICE_ID_RE`
- Mode IDs validated against `MODE_ID_RE`
- Model IDs validated against `MODEL_ID_RE`
- VRAM values validated as positive integers within GPU capacity
- Invalid IDs in LLM response **must** raise `LLMResponseError` with a user-friendly
  message — never silently accepted

Implementation: `llm/response_validator.py` parses LLM JSON output against Pydantic
models, then validates every ID field via the shared validators.

### 5.2 Prompt Template Hardening (SEC-L2)

LLM prompt templates **must** follow these rules:
- All templates stored in `llm/prompts.py` as Python constants (not user-configurable files)
- Clear instruction boundaries: system prompt separated from user data
- DB-sourced values (service names, mode names) placed in a clearly delimited data
  section, not interpolated into instruction text
- No f-string or `.format()` with user-controlled values in prompt construction
- Template variables limited to: service IDs, mode IDs, VRAM numbers, GPU capacity

Pattern:
```python
SYSTEM_PROMPT = """You are a GPU resource planner. Analyze the following GPU state
and suggest an optimal service allocation plan.

RULES:
- Only use service IDs from the PROVIDED list
- VRAM allocations must be positive integers
- Total VRAM must not exceed GPU capacity
"""

# Data section constructed separately, never interpolated into instructions
def build_user_prompt(services: list[dict], gpu_capacity: int) -> str:
    data = json.dumps({"services": services, "gpu_capacity_mb": gpu_capacity})
    return f"GPU STATE:\n{data}\n\nProvide your plan as JSON."
```

### 5.3 API Key Management (SEC-L3)

LLM API keys **must** be handled securely:
- Stored as `SecretStr` in `GpumodSettings` (pydantic-settings)
- Loaded exclusively from environment variables (`GPUMOD_LLM_API_KEY`)
- Never logged at any level (SecretStr `__repr__` returns `'**********'`)
- Never included in exception messages or error responses
- Never sent to the LLM as prompt context
- Ollama backend (local) does not require an API key

### 5.4 LLM Output Sandboxing (SEC-L4)

LLM-generated plans are **advisory only** and **never auto-executed**:
- `gpumod plan suggest` displays a table of recommendations
- Output includes suggested CLI commands the user can copy-paste
- No service start/stop/switch operations triggered by LLM output
- The plan is validated against SimulationEngine to show feasibility
- `--dry-run` flag shows the prompt without calling the LLM API

### 5.5 Data Minimization (SEC-L5)

Only the minimum necessary data is sent to external LLM APIs:
- **Sent:** Service IDs, mode IDs, VRAM allocations (MB), GPU total capacity
- **Never sent:** Database paths, file paths, API keys, environment variables,
  full service configurations, user credentials, systemd unit contents
- Ollama backend runs locally — data does not leave the machine

---

## 6. Rate Limiting (SEC-R1, SEC-R2)

### 6.1 Simulation Alternatives Cap (SEC-R1)

`SimulationEngine._generate_alternatives()` **must** cap output at **10 alternatives**.
This prevents CPU exhaustion from combinatorial exploration.

### 6.2 Request Rate Limit (SEC-R2)

The MCP server **should** implement rate limiting middleware:
- Default: 10 requests per second per connection
- Configurable via environment variable `GPUMOD_MCP_RATE_LIMIT`
- Returns clear error when limit exceeded: `{"error": "Rate limit exceeded", "code": "RATE_LIMITED"}`

---

## 7. Existing Security Controls Audit

### 7.1 Strengths (defense-in-depth)

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

### 7.2 Gaps (addressed by this spec)

| Gap | Severity | Addressed By |
|-----|----------|--------------|
| No input validation at business logic boundary | Medium | SEC-V1: MCP tools validate all args; `validation.py` module |
| Error messages may leak internal paths | Medium | SEC-E1: ErrorSanitizationMiddleware |
| No rate limiting | Low | SEC-R2: RateLimitMiddleware |
| No audit logging for mutations | Low | SEC-A1: Audit log for mutating tools |
| `_sanitize_name` not reusable outside visualization | Low | SEC-E3: Extract to `validation.py` |

---

## 8. Recommended Deployment Configuration

### 8.1 Transport

| Mode | Use Case | Security |
|------|----------|----------|
| **stdio** (default) | Claude Desktop, local LLM clients | Process-level isolation, no network exposure |
| **SSE** | Remote or multi-client | **Must** bind to `localhost` only; use reverse proxy + TLS for remote access |

### 8.2 MCP Client Configuration

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

## 9. Implementation Checklist

Tickets **must** check off relevant items before closing.

### Phase 4: Input Validation
- [x] `validation.py` module created with shared regexes and validators
- [x] All MCP tool string args validated via SEC-V1 before business logic
- [x] All MCP resource template params validated via SEC-V1
- [x] `SimulationEngine` validates IDs via SEC-V1 before DB lookups
- [x] CLI `simulate` command validates IDs (delegates to SimulationEngine)
- [x] FastMCP created with `strict_input_validation=True` (SEC-V2)

### Phase 4: Output Sanitization
- [x] `ErrorSanitizationMiddleware` strips internal paths (SEC-E1)
- [x] MCP resources contain no absolute paths (SEC-E2)
- [x] Service/mode/model names sanitized in all MCP output (SEC-E3)

### Phase 4: Authorization & Audit
- [x] Tools classified as read-only or mutating in descriptions (SEC-A1)
- [x] Mutating tools log operations at INFO level (SEC-A1)

### Phase 4: Rate Limiting & Resource Protection
- [x] Simulation alternatives capped at 10 (SEC-R1)
- [x] Rate limit middleware implemented (SEC-R2)

### Phase 4: Testing
- [x] Input validation tests: shell injection, SQL injection, template injection, path traversal
- [x] Error sanitization tests: no internal paths in error responses
- [x] Rate limit tests: excess requests rejected
- [x] Integration tests verify end-to-end security controls

### Phase 5: LLM Security Controls
- [x] LLM response IDs validated via validation.py (SEC-L1)
- [x] Prompt templates hardened with instruction boundaries (SEC-L2)
- [x] API key stored as SecretStr, never logged or in errors (SEC-L3)
- [x] LLM output advisory only, never auto-executed (SEC-L4)
- [x] Data minimization — only IDs and VRAM sent to LLM (SEC-L5)

### Phase 5: Security Hardening
- [x] RateLimitMiddleware registered on MCP server (SEC-R2)
- [x] `sanitize_name()` called in mcp_tools.py and mcp_resources.py (SEC-E3)
- [x] `visualization._sanitize_name()` replaced with import from validation.py (DRY)
- [x] `run_async()` properly typed — zero `type: ignore` in CLI modules

### Phase 5: Integration Testing
- [x] Rate limiter rejects excess MCP requests end-to-end
- [x] LLM response with injected IDs rejected (SEC-L1)
- [x] API key never appears in any error output (SEC-L3)
- [x] Config flows through CLI -> services -> DB path

### Phase 6: Security Hardening & Observability

#### Rate Limiting (SEC-R3)
- [x] Per-client rate limiting with independent quotas (SEC-R3)
- [x] Resource reads (`on_read_resource`) enforce rate limit (SEC-R2 update)

#### JSON & LLM Response Safety (SEC-P1, SEC-P2)
- [x] `safe_json_loads()` rejects payloads >1MB (SEC-P1)
- [x] `safe_json_loads()` rejects nesting depth >50 (SEC-P1)
- [x] `PlanSuggestion.reasoning` capped at 10,000 chars (SEC-P2)
- [x] Reasoning field sanitized (terminal escapes stripped) (SEC-P2)
- [x] All `json.loads()` in LLM backends replaced with `safe_json_loads()` (SEC-P1)

#### URL & Path Validation (SEC-V3, SEC-V4)
- [x] `llm_base_url` rejects `file://` scheme (SEC-V3)
- [x] `llm_base_url` rejects metadata IP ranges (169.254.x.x) (SEC-V3)
- [x] `db_path` must resolve under `$HOME` or `/tmp` (SEC-V4)

#### Sanitization & Prompt Defense (SEC-E3, SEC-L2 updates)
- [x] Mode description sanitized in MCP resource output (SEC-E3)
- [x] Service/mode names sanitized before LLM prompt JSON (SEC-L2)
- [x] `cli_mode.py` sanitizes names in Rich output (SEC-E3)

#### DB Validation (SEC-D6, SEC-V5)
- [x] `extra_config` validated against allowed key set (SEC-D6)
- [x] `vram_mb` validated with upper bound (SEC-V5)
- [x] Model IDs validated at DB boundary (SEC-V1 update)

#### HTTP Hardening (SEC-N1, SEC-N2)
- [x] Explicit per-phase timeouts on httpx clients (SEC-N1)
- [x] Response `Content-Type` validated as `application/json` (SEC-N2)
- [x] Total lifecycle timeout via `asyncio.wait_for` (SEC-N1)

#### Observability (SEC-A2, SEC-A3)
- [x] `RequestIDMiddleware` generates UUID per MCP request (SEC-A2)
- [x] Structured logging in `ServiceManager`, `Database`, `LifecycleManager` (SEC-A3)
- [x] Request ID propagates through tool call → response (SEC-A2)

#### Code Quality
- [x] `cli_context()` async context manager in `cli.py` (DRY)
- [x] All 6 CLI modules use `cli_context()` (DRY)
- [x] Zero `type: ignore` comments in `src/gpumod/` (type safety)
- [x] Dependencies pinned with upper bounds in `pyproject.toml`

#### Integration Testing
- [ ] Integration tests cover all 15 audit findings
- [ ] 900+ total tests, 97%+ coverage
