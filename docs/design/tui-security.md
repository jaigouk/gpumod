# TUI Security Design Document

> **Ticket:** gpumod-edm (P7-S2 SPIKE)
> **Status:** Complete
> **Author:** AI Architect Agent
> **Date:** 2026-02-07

---

## 1. Scope

This document investigates security implications of the Interactive TUI
(Textual-based terminal UI) described in ARCHITECTURE.md lines 475–503.
It covers terminal escape injection, Rich/Textual markup injection,
LLM plan output display, and terminal hyperlink clickjacking.

---

## 2. Threat Analysis

### 2.1 TUI-Specific Threats

| # | Threat | Vector | Impact | Mitigation | Ref |
|---|--------|--------|--------|------------|-----|
| T22 | Textual markup injection | Service/mode name containing `[bold red]` or `[link=...]` markup tags displayed via `Content.from_markup()` | Style corruption, fake status indicators, misleading colors | SEC-T1: Use `Content()` (plain text) or `Content.from_markup()` with `$variable` substitution for all user-sourced data | SEC-T1 |
| T23 | ANSI escape injection in TUI | Service name containing `\x1b[31m` sequences rendered in Textual widget | Terminal state corruption, cursor repositioning, screen clearing | SEC-T2: Apply `sanitize_name()` before display; Textual's Content class strips raw ANSI by default | SEC-T2 |
| T24 | Terminal hyperlink clickjacking | Service name containing OSC 8 hyperlink escape `\x1b]8;;http://evil.com\x1b\\Click\x1b]8;;\x1b\\` | User clicks link thinking it's legitimate | SEC-T2: `sanitize_name()` strips all control chars including ESC; Textual does not render OSC 8 from Content | SEC-T2 |
| T25 | LLM plan output injection | LLM reasoning text contains Textual markup or ANSI escapes displayed in TUI panel | Misleading status display, fake error messages | SEC-T3: Sanitize LLM reasoning before display; use `markup=False` or `Content()` for LLM output | SEC-T3 |
| T26 | Input prompt injection | TUI command input field (`> _` in mockup) accepts text that gets passed to tool calls | Command injection into MCP tools | SEC-T4: All TUI input passes through existing `validate_service_id()` / `validate_mode_id()` before tool dispatch | SEC-T4 |

### 2.2 Risk Assessment

| Threat | Likelihood | Severity | Residual Risk |
|--------|-----------|----------|---------------|
| T22 Markup injection | Medium (names stored in DB) | Low (cosmetic only) | Low — Textual `Content()` is safe by default |
| T23 ANSI injection | Medium | Medium (terminal corruption) | Low — `sanitize_name()` already strips ANSI |
| T24 Hyperlink clickjacking | Low (requires OSC 8 support) | Medium | Low — control chars stripped by `sanitize_name()` |
| T25 LLM output injection | Medium (LLM can return anything) | Medium (misleading display) | Low — sanitize + plain text rendering |
| T26 Input injection | Medium | Low (validation at tool boundary) | Low — SEC-V1 validators already enforce safe IDs |

---

## 3. Textual Framework Security Analysis

### 3.1 How Textual Handles Text

Textual (via Rich) has **two rendering modes** for text content:

1. **Markup mode** (`Content.from_markup()`, `Static(markup_string)`):
   Square brackets are interpreted as Rich markup tags. User data
   containing `[bold]`, `[red]`, or `[link=...]` will be styled.

2. **Plain text mode** (`Content(plain_string)`, `markup=False`):
   Square brackets are displayed literally. No tag interpretation.

**Key insight:** Textual does NOT auto-escape user content. If you pass
user data through `Content.from_markup()`, markup injection is possible.

### 3.2 Safe Variable Substitution

Textual provides `Content.from_markup()` with `$variable` substitution
that **automatically escapes** variable values:

```python
# SAFE: $name is auto-escaped; [bold] in name won't be interpreted
Content.from_markup("[bold]Service:[/bold] $name", name=user_provided_name)

# UNSAFE: f-string injects raw user content into markup
Content.from_markup(f"[bold]Service:[/bold] {user_provided_name}")
```

### 3.3 Recommendation for gpumod TUI

**Rule: Never use f-strings or `.format()` with `Content.from_markup()`
for user-sourced data.** Instead:

- Use `Content(text)` for purely user-sourced strings
- Use `Content.from_markup("...$var...", var=user_value)` when mixing
  markup with user data
- Use `markup=False` parameter on widgets that accept it (e.g., `notify()`)

---

## 4. Existing Defenses and How They Apply

### 4.1 `sanitize_name()` (validation.py, SEC-E3)

The existing `sanitize_name()` function provides defense-in-depth:

```python
def sanitize_name(name: str) -> str:
    # Strip ANSI escape sequences
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", name)
    # Strip Rich markup tags like [bold red]...[/bold red]
    cleaned = re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", cleaned)
    # Remove remaining control characters (except newline/tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned
```

**Coverage for TUI threats:**

| What it strips | TUI threat mitigated |
|----------------|---------------------|
| ANSI escape sequences (`\x1b[...`) | T23: ANSI injection |
| Rich markup tags (`[bold]`, `[red]`) | T22: Markup injection |
| Control characters (`\x00`–`\x1f`, `\x7f`) | T24: OSC 8 hyperlinks (ESC is `\x1b`) |

**Gap:** `sanitize_name()` does NOT strip OSC 8 hyperlink escapes in the
full format `\x1b]8;;URL\x1b\\`. However, the control char regex already
strips `\x1b` (which is `\x1b = 0x1b`), so the OSC sequence is broken.
**No gap exists.**

### 4.2 Input Validation (SEC-V1)

TUI input commands (e.g., `/switch code`, `/simulate --add svc-1`) will
parse user input and pass IDs through the existing validators:

- `validate_service_id()` — rejects anything not matching `^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$`
- `validate_mode_id()` — same regex pattern
- `validate_model_id()` — slightly broader pattern with `/` and `.`

These validators prevent shell injection, SQL injection, and path
traversal from TUI input, identical to MCP tool boundaries.

### 4.3 LLM Plan Output

LLM plan responses displayed in the TUI (e.g., reasoning text, service
suggestions) need sanitization because LLM output is untrusted:

- **reasoning field**: Already capped at 10,000 chars (SEC-P2) and
  validated by `validate_plan_response()`.
- **service_id fields**: Already validated by SEC-L1 regex.
- **Display**: Must use `Content()` (plain text) or `sanitize_name()`
  before display. Never render LLM text through `Content.from_markup()`.

---

## 5. Security Controls Summary

| Control | Description | Implementation |
|---------|-------------|----------------|
| SEC-T1 | No raw markup for user data | Use `Content()` or `$variable` substitution; never f-string into `from_markup()` |
| SEC-T2 | ANSI/control char stripping | Apply `sanitize_name()` before display; Textual Content strips raw ANSI |
| SEC-T3 | LLM output plain-text only | Render LLM reasoning via `Content()` or `markup=False`; apply `sanitize_name()` |
| SEC-T4 | TUI input validation | Route all TUI commands through SEC-V1 validators before tool dispatch |

---

## 6. Implementation Guidelines for P7-T4

### 6.1 Widget Data Flow

```
DB/API → sanitize_name() → Content() or $var substitution → Widget
                                                              ↑
LLM response → sanitize_name() → Content() ─────────────────┘
                                                              ↑
User input → validate_*_id() → command dispatch ─────────────┘
```

### 6.2 Code Patterns

#### Service List Widget

```python
from textual.content import Content

class ServiceList(Widget):
    def render_service(self, service: Service) -> Content:
        # SAFE: $name is auto-escaped by Content.from_markup()
        state_color = STATE_COLORS.get(service.state, "dim")
        return Content.from_markup(
            "[$color]●[/$color] $name  $vram",
            color=state_color,
            name=sanitize_name(service.name),
            vram=f"{service.vram_mb}MB",
        )
```

#### LLM Plan Display

```python
class PlanPanel(Widget):
    def render_reasoning(self, reasoning: str) -> Content:
        # SAFE: Content() treats text as plain — no markup interpretation
        return Content(sanitize_name(reasoning))
```

#### Command Input Processing

```python
class CommandInput(Input):
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text.startswith("/switch "):
            mode_id = text[8:].strip()
            try:
                validate_mode_id(mode_id)
            except ValueError:
                self.app.notify("Invalid mode ID", severity="error")
                return
            await self.app.switch_mode(mode_id)
```

### 6.3 Testing Requirements

| Test | Description | Control |
|------|-------------|---------|
| `test_service_name_markup_not_rendered` | Name with `[bold]` displays literally | SEC-T1 |
| `test_ansi_stripped_before_display` | Name with `\x1b[31m` is cleaned | SEC-T2 |
| `test_llm_reasoning_plain_text` | LLM output with markup renders as plain | SEC-T3 |
| `test_tui_input_validates_ids` | Invalid IDs rejected before dispatch | SEC-T4 |
| `test_osc8_hyperlink_stripped` | OSC 8 link sequence broken by sanitize | SEC-T2 |

---

## 7. Conclusion

The gpumod TUI has **low residual security risk** because:

1. **Existing `sanitize_name()`** already strips ANSI escapes, Rich
   markup tags, and control characters — covering T22, T23, T24.

2. **Textual's `Content()` class** treats text as plain by default;
   only `Content.from_markup()` interprets markup, and it supports
   safe `$variable` substitution that auto-escapes.

3. **Existing SEC-V1 validators** cover TUI input at the same boundary
   as MCP tools — no new validation needed.

4. **LLM output** is already length-capped and ID-validated; displaying
   via `Content()` (plain text) prevents markup injection.

**No new security controls are needed** beyond applying the existing
defenses consistently. The implementation ticket (P7-T4) should follow
the code patterns in Section 6.2 and add the tests in Section 6.3.

---

## 8. Out of Scope

- Textual Web deployment security (authentication, TLS) — not planned
- Clipboard injection via terminal paste — OS-level concern
- Screen scraping via terminal access — physical security concern
- Multi-user TUI sessions — gpumod is single-user
