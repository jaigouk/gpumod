# gpumod Documentation Update: macOS Apple Silicon Platform Support

> **Ticket:** (to be assigned)
> **Type:** task
> **Priority:** P2
> **Epic:** gpumod-dwm (macOS Apple Silicon support)
> **Replaces:** gpumod-6rc (was too narrow -- only covered architecture doc)
> **Depends on:** ALL implementation tickets #1-#11 confirmed working
> **Depends on:** gpumod-zxx (pre-implementation architecture update) -- completed first
> **No TDD required** -- documentation only, no code changes

---

## Goal / Problem

After all macOS Apple Silicon implementation tickets (#1-#11) have shipped,
every user-facing document still describes gpumod as a Linux-only, NVIDIA-only,
systemd-only tool. Users on macOS will find incorrect install instructions,
wrong CLI output examples, missing platform prerequisites, and references to
tools (`nvidia-smi`, `systemctl`, `journalctl`, `loginctl`) that do not exist
on their system.

This ticket is the **final verification pass** ensuring all user-facing docs
match the shipped dual-platform code. It runs AFTER implementation is
confirmed working, not before.

---

## Background / Context: File-by-File Audit

### 1. `docs/index.md` -- Website Landing Page

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 3 | `description: ... on NVIDIA GPUs with VRAM simulation...` | NVIDIA-only in meta description |
| 15 | `GPU Service Manager for ML workloads on Linux/NVIDIA systems.` | Linux/NVIDIA-only positioning |
| 17-19 | `gpumod manages ... on NVIDIA GPUs. It tracks VRAM allocation...` | NVIDIA-only framing |
| 35 | `Template Engine -- Generate and install systemd unit files` | systemd-only |
| 55 | `# Deploy a service (auto-generates systemd unit file)` | systemd-only comment |
| 97 | `- Linux with NVIDIA GPU` | Linux-only requirement |
| 98 | `- nvidia-smi in PATH` | nvidia-smi-only requirement |

**What needs to change:** Platform-neutral language ("GPU services on Linux
and macOS"), mention both systemd and launchd, list both NVIDIA and Apple
Silicon as supported platforms in requirements.

### 2. `docs/getting-started/index.md` -- Getting Started Landing

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 3 | `description: ... on Linux/NVIDIA systems.` | Linux-only meta |
| 12 | `- Linux with NVIDIA GPU and nvidia-smi in PATH` | Linux-only requirement |
| 69-70 | `GPU hardware profile (detected from nvidia-smi)` | nvidia-smi-only |
| 85 | `gpumod auto-generates systemd unit files from presets` | systemd-only |
| 88-91 | `Preview the generated unit file` section | systemd-only |
| 95-103 | Full section on `systemctl --user`, `loginctl enable-linger` | Linux-specific workflow |
| 106-109 | `gpumod template install ... writes to ~/.config/systemd/user/` | systemd-only path |
| 113-116 | `systemctl --user daemon-reload` | Linux-only command |

**What needs to change:** Add platform-conditional sections. Show macOS
prerequisites (Xcode CLT, Homebrew). Show launchd equivalents alongside
systemd instructions. Update GPU detection to mention `ioreg` / Metal.
Replace `loginctl enable-linger` section with macOS equivalent (launchd
runs automatically for user agents). Replace `systemctl --user daemon-reload`
with macOS `launchctl` equivalent.

### 3. `docs/getting-started/cli.md` -- CLI Reference

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 134 | `Manage Jinja2 systemd unit file templates.` | systemd-only |
| 155-156 | `Generate a systemd unit file for a registered service.` | systemd-only |
| 170-171 | `Install ... to the systemd directory. Requires --yes.` | systemd-only |
| 182 | `~/.config/systemd/user/gpumod-{service_id}.service` | Linux path |
| 439-454 | `gpumod install-server` section: systemd-specific throughout | systemd-only install |
| 456-461 | `systemctl --user enable/start gpumod-mcp` | Linux-only commands |

**What needs to change:** `template generate` and `template install` should
mention that on macOS, plist files are generated instead of systemd units.
The install path on macOS is `~/Library/LaunchAgents/com.gpumod.*.plist`.
The `install-server` section needs macOS equivalents (`launchctl` commands).

### 4. `docs/getting-started/configuration.md` -- Configuration Guide

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 25 | `GPUMOD_VRAM_SAFETY_MARGIN_MB` | May need clarification for unified memory |

**What needs to change:** Minimal changes. Add a note explaining that on macOS
Apple Silicon, "VRAM" refers to the GPU-accessible portion of unified memory.
The `VRAM_SAFETY_MARGIN_MB` env var applies the same way. Add any new
macOS-specific env vars introduced by implementation tickets (e.g.,
`GPUMOD_PLATFORM` override if one was added).

### 5. `docs/gpumod-5min.md` -- 5-Minute Presentation (Marp slides)

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 297-318 | Problem slide: `24GB`, `CUDA OOM`, NVIDIA GPU assumptions | NVIDIA-specific examples |
| 345 | `systemd units and doing VRAM math` | systemd reference |
| 549 | `Open Source. Python. Apache 2.0.` | Could note cross-platform |

**What needs to change:** Low priority. The presentation is a demo artifact,
not primary docs. Consider adding a slide or note that gpumod works on both
Linux/NVIDIA and macOS/Apple Silicon. Replace "CUDA OOM" with a more general
"GPU out-of-memory" or keep it as a concrete Linux example and add a macOS
equivalent. The `systemd units` reference in speaker notes can mention "or
launchd plists" in parentheses.

### 6. `docs/contributing.md` -- Contributing Guide

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 33 | `# Run E2E tests on GPU machines` | Should clarify both platforms |

**What needs to change:** Add a note that E2E tests can run on both Linux
(with NVIDIA GPU) and macOS (with Apple Silicon). Mention any new pytest
markers (e.g., `@pytest.mark.macos_required`, `@pytest.mark.linux_required`)
if introduced by implementation tickets. Add macOS dev setup prerequisites
if different from Linux (e.g., `brew install` instead of `apt install`).

### 7. `docs/user-guide/mcp-workflows.md` -- MCP Workflow Docs

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 38 | `{"vram_free_mb": 24408, ...}` | 24GB NVIDIA GPU example |
| 370 | `journalctl --user -u gpumod-glm-code.service -n 50` | systemd-only log command |

**What needs to change:** The VRAM examples can stay as-is (they are
illustrative). The `journalctl` troubleshooting command needs a macOS
equivalent showing how to read launchd service logs (e.g.,
`log show --predicate 'subsystem == "com.gpumod"' --last 5m` or reading
the `StandardOutPath` log file).

### 8. `docs/user-guide/mcp.md` -- MCP General Docs

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 65 | `~/.config/claude/claude_desktop_config.json (Linux)` | Already mentions both Linux and macOS |
| 66 | `~/Library/Application Support/Claude/... (macOS)` | Already has macOS path |

**What needs to change:** Minimal. This file already shows both Linux and
macOS config paths for Claude Desktop (line 65-66). Verify the `install-server`
references match the updated CLI docs. No systemd-specific language detected.

### 9. `docs/user-guide/ai-planning.md` -- AI Planning Guide

No Linux/systemd/NVIDIA-specific language detected. The file discusses
LLM-assisted VRAM planning which is platform-agnostic.

**What needs to change:** Nothing. Verify after implementation that "VRAM"
terminology is still accurate for unified memory systems.

### 10. `docs/architecture/index.md` -- Architecture (arc42)

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 3 | `description: ... on single-GPU Linux systems.` | Linux-only meta |
| 12 | `on single-GPU Linux systems` | Linux-only |
| 40-43 | Constraints table: `Linux only`, `Single GPU`, `NVIDIA only` | Hard constraints |
| 63-66 | System context: `systemd (service lifecycle)`, `nvidia-smi (GPU queries)` | Linux-specific |
| 114 | `systemd user services` | Process control |
| 115 | `nvidia-smi polling` | VRAM tracking |
| 160-173 | System layer: `systemd: unit management`, `nvidia-smi: GPU info` | Linux-specific |
| 260 | `Query nvidia-smi for current usage` | VRAM tracking |
| 288-293 | Driver table: all show `systemd` for process control | Linux-specific |
| 327-339 | Mode switch sequence: `systemctl stop/start` | systemd-specific |
| 376-398 | Deployment view: `~/.config/systemd/user/`, `systemd --user` | Linux paths |
| 484 | `Wait for VRAM release ... poll nvidia-smi` | nvidia-smi |
| 639-651 | ADR-3: `systemd User Services`, `(-) Linux-only (no macOS/Windows)` | Documents the Linux limitation |

**What needs to change:** This should already be updated by gpumod-zxx
(pre-implementation architecture update). This ticket verifies the
architecture doc matches shipped code:
- Constraints table updated to show both platforms
- System context diagram shows ProcessController abstraction
- Driver table shows both systemd and launchd for process control
- Mode switch sequence is generic (ProcessController, not systemctl)
- Deployment view shows both platform paths
- ADR-3 updated or supplemented with ADR for platform abstraction
- New ADR referencing the macOS spike findings

### 11. `docs/architecture/SECURITY.md` -- Security Considerations

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 24 | `reaches systemctl` | systemd-specific threat vector |
| 128-130 | `Starts/stops systemd units` | systemd-specific side effects |
| 281 | `systemctl command allowlist` | systemd-specific control |

**What needs to change:** Security controls need to mention both init
systems. The command allowlist in `SystemdController` stays, but a parallel
note about `LaunchdController` command validation should be added. Threat T1
should mention both `systemctl` and `launchctl` as injection targets. The
tool classification table should note that on macOS, mutating tools
start/stop launchd agents instead of systemd units.

### 12. `docs/design/docker-driver.md` -- Docker Driver Design

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 45 | `systemd.py (line 85-89)` reference | Internal reference |
| 84-85 | `Docker socket ... /var/run/docker.sock` | Linux-specific path |
| 339 | `ServiceRegistry` integration with `VLLMDriver` etc. | Lists all drivers |

**What needs to change:** Add a note that Docker Desktop on macOS uses a
different socket path (`~/.docker/run/docker.sock` or
`/var/run/docker.sock` symlinked by Docker Desktop). The design is otherwise
platform-agnostic since the Docker SDK abstracts the socket. Add note about
GPU passthrough differences (Linux: `--gpus` / `nvidia` runtime; macOS:
Docker Desktop does not support GPU passthrough to containers).

### 13. `docs/internal/presets.md` -- Preset Format

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 43 | `unit_vars: Variables passed to the systemd template` | systemd-only |
| 50-55 | Driver table: template column shows `.service.j2` | systemd templates |
| 99 | `unit_vars:` in vLLM example | Naming implies systemd |
| 143 | `working_dir: /opt/embedding` | Linux path in example |

**What needs to change:** Note that `unit_vars` are passed to the
platform-appropriate template (systemd `.service.j2` on Linux, launchd
`.plist.j2` on macOS). Update the driver table to show both template
types. Mention that `unit_template` can specify a platform-specific
override. Example paths should use generic or `~`-based paths.

### 14. `docs/internal/presets-workflow.md` -- Preset Workflow

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 9 | `cudaMalloc failed: out of memory` | CUDA-specific error |
| 57 | `nvidia-smi --query-gpu=memory.free,...` | nvidia-smi command |
| 150 | `watch -n 1 nvidia-smi` | nvidia-smi monitoring |
| 196 | `cudaMalloc failed: out of memory` (troubleshooting) | CUDA-specific |

**What needs to change:** This is an internal doc but referenced by
contributors. Add macOS equivalents: `ioreg -r -d 1 -w 0 -c IOGPUDevice`
for GPU memory queries, `sudo powermetrics --samplers gpu_power -i 1000`
for monitoring. Change "cudaMalloc failed" to a platform-conditional
note (CUDA OOM on Linux, Metal allocation failure on macOS).

### 15. `.claude/CLAUDE.md` -- Project Instructions

| Line(s) | Current text | Issue |
|----------|-------------|-------|
| 5 | `on NVIDIA GPUs via systemd user units` | NVIDIA + systemd only |
| 22 | `Systemd templates: src/gpumod/templates/systemd/*.j2` | systemd-only |
| 99 | `Jinja2 sandboxed templates for systemd unit generation` | systemd-only |
| 101-102 | `User-level systemd units (WantedBy=default.target, no User= directive)` | systemd-specific convention |
| 102 | `All templates must include StartLimitBurst and StartLimitIntervalSec` | systemd-specific convention |

**What needs to change:** This should already be updated by gpumod-zxx
(pre-implementation). This ticket verifies it matches shipped code:
- Project description mentions both platforms
- Architecture section lists both template directories
- Conventions mention both systemd and launchd template requirements
- Template convention for launchd (e.g., `ThrottleInterval`,
  `KeepAlive` requirements) added alongside systemd conventions

---

## Design

### Organizing Principle

Updates are organized by doc file. Within each file, changes follow a
consistent pattern:

1. **Platform-neutral language** -- Replace "Linux" with "Linux and macOS",
   "NVIDIA GPU" with "NVIDIA or Apple Silicon GPU", "systemd" with "systemd
   (Linux) / launchd (macOS)".

2. **Tabbed examples** -- Where platform-specific commands or paths appear,
   use MkDocs Material's content tabs:
   ```markdown
   === "Linux"
       ```bash
       systemctl --user start gpumod-vllm-chat
       ```
   === "macOS"
       ```bash
       launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gpumod.vllm-chat.plist
       ```
   ```

3. **Requirements sections** -- Show both platforms with clear labels:
   ```markdown
   **Linux:**
   - NVIDIA GPU with `nvidia-smi` in PATH
   - systemd with user session (`loginctl enable-linger`)

   **macOS:**
   - Apple Silicon Mac (M1 or later)
   - macOS 13+ (Ventura or later)
   ```

4. **Preserve existing content** -- Do not delete Linux/NVIDIA examples.
   Add macOS alongside them.

### Content Source for macOS Specifics

All macOS-specific content must come from the verified spike findings:
- `docs/internal/research/20260330_macos_spike/` -- verified scripts and results
- `docs/internal/research/macos-gpu-memory-launchd.md` -- launchd/GPU memory research
- `docs/internal/research/20260330_macos_spike/ADR-platform-abstraction.md` -- architecture decisions
- Shipped source code in `src/gpumod/` -- the ground truth for what was implemented

Do NOT invent macOS CLI output or behavior. All examples must match the
actual shipped implementation.

---

## Steps (Ordered by Priority)

### Phase 1: User-Facing Docs (highest impact)

1. **`docs/index.md`** -- Update landing page positioning, requirements,
   quick start code block. First thing new users see.

2. **`docs/getting-started/index.md`** -- Update requirements, installation,
   first-time setup, deploying a service sections. The primary onboarding path.

3. **`docs/getting-started/cli.md`** -- Update template commands, install-server
   section, all systemd-specific CLI descriptions.

4. **`docs/getting-started/configuration.md`** -- Add unified memory note,
   any new macOS env vars.

5. **`docs/gpumod-5min.md`** -- Light touch: add macOS mention in a slide or
   speaker notes.

### Phase 2: User Guide Docs

6. **`docs/user-guide/mcp-workflows.md`** -- Add macOS log command in
   troubleshooting workflow.

7. **`docs/user-guide/mcp.md`** -- Verify `install-server` references are
   consistent with updated CLI docs.

8. **`docs/user-guide/ai-planning.md`** -- Verify only; likely no changes.

### Phase 3: Architecture and Security

9. **`docs/architecture/index.md`** -- Verify gpumod-zxx updates match shipped
   code. Fill any gaps.

10. **`docs/architecture/SECURITY.md`** -- Add launchd security controls,
    update threat model.

### Phase 4: Internal and Design Docs

11. **`docs/design/docker-driver.md`** -- Add macOS Docker socket and GPU
    passthrough notes.

12. **`docs/internal/presets.md`** -- Update template references, add macOS
    template column.

13. **`docs/internal/presets-workflow.md`** -- Add macOS GPU monitoring commands
    alongside nvidia-smi.

### Phase 5: Project Meta

14. **`.claude/CLAUDE.md`** -- Verify gpumod-zxx updates match shipped code.
    Fill any gaps.

15. **`docs/contributing.md`** -- Add macOS dev setup, test markers.

---

## Acceptance Criteria

### Per-File Checklist

- [ ] **`docs/index.md`**
  - [ ] Meta description mentions both platforms
  - [ ] "Linux/NVIDIA systems" replaced with platform-neutral text
  - [ ] Requirements section lists both Linux and macOS prerequisites
  - [ ] Quick start code block works on both platforms or shows tabs
  - [ ] "systemd unit file" language updated to be platform-aware

- [ ] **`docs/getting-started/index.md`**
  - [ ] Requirements section shows both platforms
  - [ ] `nvidia-smi` detection note includes macOS alternative
  - [ ] "systemd unit files" section shows both platforms
  - [ ] `loginctl enable-linger` section has macOS equivalent or note
  - [ ] `systemctl --user daemon-reload` has macOS equivalent
  - [ ] Install paths show both `~/.config/systemd/user/` and `~/Library/LaunchAgents/`

- [ ] **`docs/getting-started/cli.md`**
  - [ ] `template` subcommand docs mention plist generation on macOS
  - [ ] `template install` shows macOS install path
  - [ ] `install-server` section shows macOS launchd alternative
  - [ ] `systemctl --user enable/start` has `launchctl` equivalent

- [ ] **`docs/getting-started/configuration.md`**
  - [ ] Unified memory note added for Apple Silicon
  - [ ] Any new macOS env vars documented
  - [ ] VRAM safety margin explained for unified memory

- [ ] **`docs/gpumod-5min.md`**
  - [ ] At least one mention of cross-platform support
  - [ ] Speaker notes updated if referencing systemd

- [ ] **`docs/contributing.md`**
  - [ ] macOS dev setup mentioned alongside Linux
  - [ ] New test markers documented if any
  - [ ] E2E test instructions cover both platforms

- [ ] **`docs/user-guide/mcp-workflows.md`**
  - [ ] `journalctl` command has macOS log equivalent
  - [ ] No Linux-only troubleshooting steps without macOS alternative

- [ ] **`docs/user-guide/mcp.md`**
  - [ ] `install-server` references consistent with CLI docs
  - [ ] Claude Desktop config paths still show both platforms

- [ ] **`docs/user-guide/ai-planning.md`**
  - [ ] Verified: no platform-specific changes needed (or minimal)

- [ ] **`docs/architecture/index.md`**
  - [ ] Verified: gpumod-zxx changes match shipped code
  - [ ] Constraints table shows both platforms
  - [ ] System context diagram shows ProcessController
  - [ ] ADR-3 updated or supplemented for platform abstraction
  - [ ] No stale "Linux-only" or "NVIDIA-only" language remains

- [ ] **`docs/architecture/SECURITY.md`**
  - [ ] Threat T1 mentions both systemctl and launchctl
  - [ ] Tool classification notes both init systems
  - [ ] Defense-in-depth table includes launchd controls
  - [ ] No security controls described only for one platform

- [ ] **`docs/design/docker-driver.md`**
  - [ ] macOS Docker socket path noted
  - [ ] GPU passthrough limitation on macOS documented

- [ ] **`docs/internal/presets.md`**
  - [ ] `unit_vars` description mentions both template types
  - [ ] Driver table shows both systemd and launchd templates
  - [ ] Example paths are generic (not Linux-specific)

- [ ] **`docs/internal/presets-workflow.md`**
  - [ ] macOS GPU monitoring commands added
  - [ ] CUDA OOM error has macOS equivalent noted
  - [ ] nvidia-smi commands have ioreg/Metal equivalents

- [ ] **`.claude/CLAUDE.md`**
  - [ ] Project description mentions both platforms
  - [ ] Template paths include both systemd and launchd
  - [ ] Conventions list both platform template requirements

### Global Checks

- [ ] No doc file contains "Linux only" without mentioning macOS
- [ ] No doc file lists `nvidia-smi` as a requirement without macOS alternative
- [ ] No doc file shows `systemctl` commands without macOS equivalent
- [ ] All macOS examples verified against shipped implementation
- [ ] `mkdocs build` completes without errors
- [ ] All internal links still resolve
- [ ] No PII in any committed doc file (per Privacy section in CLAUDE.md)

---

## Risks / Dependencies

### Hard Dependencies

| Dependency | Why |
|-----------|-----|
| Implementation tickets #1-#11 ALL complete and verified | Cannot document behavior that is not yet shipped |
| gpumod-zxx (architecture doc update) complete | Architecture doc should be updated before this general pass |
| macOS spike (gpumod-2qc) findings verified | All macOS content sourced from verified research |

### Risks

| Risk | Mitigation |
|------|-----------|
| Implementation changes after docs are written | Run this ticket LAST, after all implementation PRs are merged |
| Incorrect macOS examples | All examples must be verified against running code on macOS |
| MkDocs Material tab syntax breaks | Test with `mkdocs serve` before merging |
| Over-updating internal docs | Internal docs (Phase 4) are lower priority; focus on user-facing first |
| Scope creep into code changes | This ticket is docs-only. Any code issues found become separate bug tickets |

### Out of Scope

- Code changes (separate tickets)
- New documentation pages (only update existing)
- Windows support documentation
- Benchmark docs (platform-specific by nature, not user-facing setup docs)
- Internal QA reports (historical records, not updated retroactively)
- Research docs under `docs/internal/research/` (these are spike artifacts)
