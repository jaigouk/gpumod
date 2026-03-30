# gpumod-2gp: Create launchd .plist.j2 Jinja2 Templates

**Type:** Feature
**Priority:** P2
**Depends on:** gpumod-khc (Template engine platform dispatch)
**Parent epic:** gpumod-dwm (macOS Apple Silicon support)
**ADR reference:** ADR-platform-abstraction.md, follow-up ticket #7 (lines 418-420)

---

## Goal / Problem

gpumod currently generates systemd `.service` unit files from Jinja2 templates
for Linux. macOS uses launchd instead of systemd. To support Apple Silicon,
gpumod needs launchd `.plist` equivalents of the four existing systemd
templates so that services can be managed via `launchctl` on macOS.

This ticket creates four new Jinja2 templates in `src/gpumod/templates/launchd/`:

- `vllm.plist.j2`
- `llamacpp.plist.j2`
- `fastapi.plist.j2`
- `mcp-server.plist.j2`

Each template renders valid Apple plist XML that is functionally equivalent to
its systemd counterpart, using the **same Jinja2 variable names** so the
platform-dispatched `TemplateEngine` (ticket gpumod-khc) can render either
format transparently.

---

## Background / Context

### Systemd-to-launchd Feature Mapping

Source: `macos-gpu-memory-launchd.md` lines 296-316 (Section 3, Feature Mapping table).

| systemd Directive | Plist Key | Notes | Source |
|---|---|---|---|
| `[Unit] Description=` | `<!-- XML comment -->` | Plist has no description field; use XML comment | -- |
| `[Unit] After=network.target` | *(none)* | launchd has no dependency ordering (`macos-gpu-memory-launchd.md:319`). gpumod handles ordering in LifecycleManager | -- |
| `[Unit] StartLimitBurst=5` / `StartLimitIntervalSec=60` | Built-in 10-second crash rule + `ThrottleInterval` | launchd's crash detection is hardcoded: exit within 10s of launch = crash (`macos-gpu-memory-launchd.md:304,322`). `ThrottleInterval` paces restarts. No direct burst-count equivalent. | ADR line 386 |
| `[Service] Type=simple` | *(implicit)* | launchd does not have service types; all processes are simple | `macos-gpu-memory-launchd.md:311` |
| `[Service] ExecStart=...` | `<key>ProgramArguments</key><array>` | Must be an array of strings, not a single command string (`macos-gpu-memory-launchd.md:300`) | -- |
| `[Service] Environment="K=V"` | `<key>EnvironmentVariables</key><dict>` | Plist dictionary with key/string pairs (`macos-gpu-memory-launchd.md:305`) | -- |
| `[Service] WorkingDirectory=` | `<key>WorkingDirectory</key><string>` | Direct equivalent | -- |
| `[Service] Restart=on-failure` | `<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>` | Restart only on non-zero exit (`macos-gpu-memory-launchd.md:302`). Per Q9 findings: crash-only restart is correct for ML servers. | -- |
| `[Service] RestartSec=30` | `<key>ThrottleInterval</key><integer>30</integer>` | Min seconds between relaunch attempts (`macos-gpu-memory-launchd.md:303`). Per Q8: launchd tolerates slow starts. | ADR line 265, 386 |
| `[Service] KillMode=control-group` | *(not needed)* | launchd kills the process tree by default on `bootout` | -- |
| `[Service] TimeoutStopSec=30` | `<key>ExitTimeOut</key><integer>30</integer>` | Seconds before SIGKILL after SIGTERM | -- |
| `[Service] StandardOutput=journal` | `<key>StandardOutPath</key><string>` | launchd writes to files, not journal (`macos-gpu-memory-launchd.md:306-307`) | -- |
| `[Service] StandardError=journal` | `<key>StandardErrorPath</key><string>` | Same as above | -- |
| `[Install] WantedBy=default.target` | `<key>RunAtLoad</key><false/>` | We use `false` because gpumod manages lifecycle explicitly via `launchctl bootstrap`; services should not auto-start on login. The systemd equivalent `WantedBy=default.target` enables the unit but gpumod still controls start/stop. On macOS, `RunAtLoad=true` would start the service immediately when the plist is loaded, which conflicts with gpumod's mode-based orchestration. | `macos-gpu-memory-launchd.md:308` |
| *(no equivalent)* | `<key>ProcessType</key><string>Background</string>` | Tells macOS this is a background process; affects scheduling priority and thermal throttling. Required for ML workloads. | Apple launchd docs |

### Template Variable Interface

All four service templates (vllm, llamacpp, fastapi) receive the same context
from `render_service_unit()` at `engine.py:108-146`:

```python
context = {
    "service": service,      # gpumod.models.Service (id, name, driver, port, model_id, ...)
    "settings": settings,    # dict[str, str] (cuda_devices, hf_home, vllm_bin, etc.)
    "unit_vars": unit_vars,  # dict[str, Any] (driver-specific: gpu_mem_util, model_path, etc.)
    "extra_env": extra_env,  # dict[str, str] (additional environment variables)
}
```

The `mcp-server.service.j2` template is rendered separately by
`mcp_installer.py:52-102` (`render_mcp_unit()`) with different variables:

```python
context = {
    "python_bin": str,     # abs path to Python interpreter
    "venv_bin": str,       # abs path to venv bin directory
    "working_dir": str,    # abs path to project root
    "transport": str,      # default "streamable-http"
    "host": str,           # default "127.0.0.1"
    "port": int,           # default 8808
}
```

The launchd templates must use these **exact same variable names** so that
no changes to the rendering call sites are needed (beyond what ticket
gpumod-khc introduces for platform dispatch).

### Log Path Convention

Launchd services write stdout/stderr to files (not a journal). Convention:

```
~/Library/Logs/gpumod/<service-name>.stdout.log
~/Library/Logs/gpumod/<service-name>.stderr.log
```

Templates must not hardcode home directory paths. Use Jinja2 variables
(`settings.log_dir` or a derived path) per `.claude/CLAUDE.md:10-16`
(Privacy & Open Source requirements).

Since `settings` is a plain `dict[str, str]`, the log directory must be
passed as `settings.log_dir` by the caller. The templates should use:

```xml
<key>StandardOutPath</key>
<string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stdout.log</string>
```

For the mcp-server template (different variable namespace):

```xml
<key>StandardOutPath</key>
<string>{{ log_dir | default('~/Library/Logs/gpumod') }}/gpumod-mcp.stdout.log</string>
```

### Label Convention

Per ADR line 66, `LaunchdController` validates against the `com.gpumod.<name>`
convention. Labels in plist templates must follow:

```
com.gpumod.{{ service.name }}     # for vllm, llamacpp, fastapi
com.gpumod.mcp-server             # for mcp-server (fixed)
```

---

## Design

### Template Directory Layout

Per ADR lines 106-118:

```
src/gpumod/templates/
  systemd/                    # existing -- unchanged
    vllm.service.j2
    llamacpp.service.j2
    fastapi.service.j2
    mcp-server.service.j2
  launchd/                    # NEW
    vllm.plist.j2
    llamacpp.plist.j2
    fastapi.plist.j2
    mcp-server.plist.j2
```

The `_SAFE_NAME_RE` pattern at `engine.py:27`
(`^[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_.]+$`) already accepts `.plist.j2` filenames
(ADR line 121-123). No regex change needed.

### 1. vllm.plist.j2

Reference: `systemd/vllm.service.j2` (lines 1-37).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- {{ service.name }} -->
    <key>Label</key>
    <string>com.gpumod.{{ service.name }}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{{ settings.vllm_bin | default('vllm') }}</string>
        <string>serve</string>
        <string>{{ service.model_id }}</string>
        <string>--port</string>
        <string>{{ service.port }}</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--gpu-memory-utilization</string>
        <string>{{ unit_vars.gpu_mem_util | default(0.9) }}</string>
        <string>--max-model-len</string>
        <string>{{ unit_vars.max_model_len | default(4096) }}</string>
{%- if unit_vars.max_num_seqs is defined %}
        <string>--max-num-seqs</string>
        <string>{{ unit_vars.max_num_seqs }}</string>
{%- endif %}
{%- if unit_vars.dtype is defined %}
        <string>--dtype</string>
        <string>{{ unit_vars.dtype }}</string>
{%- endif %}
{%- if unit_vars.enforce_eager | default(false) %}
        <string>--enforce-eager</string>
{%- endif %}
{%- if unit_vars.runner is defined %}
        <string>--task</string>
        <string>{{ 'embed' if unit_vars.runner == 'pooling' else unit_vars.runner }}</string>
{%- endif %}
{%- if unit_vars.hf_overrides is defined %}
        <string>--hf-overrides</string>
        <string>{{ unit_vars.hf_overrides }}</string>
{%- endif %}
{%- if unit_vars.enable_sleep_mode | default(false) %}
        <string>--enable-sleep-mode</string>
{%- endif %}
{%- if unit_vars.sleep_level is defined %}
        <string>--sleep-level</string>
        <string>{{ unit_vars.sleep_level }}</string>
{%- endif %}
{%- if unit_vars.trust_remote_code | default(false) %}
        <string>--trust-remote-code</string>
{%- endif %}
{%- if unit_vars.extra_args is defined %}
{%-   for arg in unit_vars.extra_args.split() %}
        <string>{{ arg }}</string>
{%-   endfor %}
{%- endif %}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CUDA_VISIBLE_DEVICES</key>
        <string>{{ settings.cuda_devices | default('0') }}</string>
        <key>HF_HOME</key>
        <string>{{ settings.hf_home | default('~/.cache/huggingface') }}</string>
{%- for key, value in extra_env.items() %}
        <key>{{ key }}</key>
        <string>{{ value }}</string>
{%- endfor %}
    </dict>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ExitTimeOut</key>
    <integer>30</integer>

    <key>ProcessType</key>
    <string>Background</string>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stderr.log</string>
</dict>
</plist>
```

**Directive mapping from `vllm.service.j2`:**

| systemd (vllm.service.j2 line) | Plist key | Notes |
|---|---|---|
| `Description={{ service.name }}` (line 2) | `<!-- {{ service.name }} -->` comment | Plist has no description field |
| `After=network.target` (line 3) | *(omitted)* | No launchd equivalent (`macos-gpu-memory-launchd.md:319`) |
| `StartLimitBurst=5` (line 4) | `ThrottleInterval=30` | launchd uses throttle + built-in 10s rule instead (`macos-gpu-memory-launchd.md:304`) |
| `StartLimitIntervalSec=60` (line 5) | *(covered by ThrottleInterval)* | -- |
| `Type=simple` (line 8) | *(implicit)* | All launchd jobs are simple (`macos-gpu-memory-launchd.md:311`) |
| `Environment="CUDA_VISIBLE_DEVICES=..."` (line 9) | `EnvironmentVariables` dict | `macos-gpu-memory-launchd.md:305` |
| `Environment="HF_HOME=..."` (line 10) | Inside same dict | -- |
| `{% for %} Environment=...` (line 11-13) | Loop inside dict | Same `extra_env` variable |
| `ExecStart=...` (lines 15-29) | `ProgramArguments` array | Each arg is a separate `<string>` element (`macos-gpu-memory-launchd.md:300`) |
| `Restart=on-failure` (line 31) | `KeepAlive.SuccessfulExit=false` | Restart on crash only (`macos-gpu-memory-launchd.md:302`) |
| `RestartSec=30` (line 32) | `ThrottleInterval=30` | `macos-gpu-memory-launchd.md:303` |
| `KillMode=control-group` (line 33) | *(default behavior)* | launchd kills process tree on bootout |
| `TimeoutStopSec=30` (line 34) | `ExitTimeOut=30` | Seconds before SIGKILL |
| `WantedBy=default.target` (line 37) | `RunAtLoad=false` | gpumod manages lifecycle |

### 2. llamacpp.plist.j2

Reference: `systemd/llamacpp.service.j2` (lines 1-49).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- {{ service.name }} -->
    <key>Label</key>
    <string>com.gpumod.{{ service.name }}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{{ settings.llamacpp_bin | default('llama-server') }}</string>
{%- if unit_vars.models_dir is defined %}
        <string>--models-dir</string>
        <string>{{ unit_vars.models_dir }}</string>
{%-   if unit_vars.no_models_autoload | default(false) %}
        <string>--no-models-autoload</string>
{%-   endif %}
{%-   if unit_vars.models_max is defined %}
        <string>--models-max</string>
        <string>{{ unit_vars.models_max }}</string>
{%-   endif %}
{%- else %}
        <string>--model</string>
        <string>{{ unit_vars.model_path }}</string>
{%- endif %}
        <string>--port</string>
        <string>{{ service.port }}</string>
        <string>--host</string>
        <string>{{ unit_vars.host | default('127.0.0.1') }}</string>
{%- if unit_vars.context_size is defined or unit_vars.models_dir is not defined %}
        <string>--ctx-size</string>
        <string>{{ unit_vars.context_size | default(4096) }}</string>
{%- endif %}
        <string>--n-gpu-layers</string>
        <string>{{ unit_vars.n_gpu_layers | default(-1) }}</string>
{%- if unit_vars.jinja | default(false) %}
        <string>--jinja</string>
{%- endif %}
{%- if unit_vars.flash_attn | default(false) %}
        <string>--flash-attn</string>
{%- endif %}
{%- if unit_vars.preset_file is defined %}
        <string>--override-kv</string>
        <string>{{ unit_vars.preset_file }}</string>
{%- endif %}
{%- if unit_vars.extra_args is defined %}
{%-   for arg in unit_vars.extra_args.split() %}
        <string>{{ arg }}</string>
{%-   endfor %}
{%- endif %}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CUDA_VISIBLE_DEVICES</key>
        <string>{{ settings.cuda_devices | default('0') }}</string>
{%- for key, value in extra_env.items() %}
        <key>{{ key }}</key>
        <string>{{ value }}</string>
{%- endfor %}
    </dict>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ProcessType</key>
    <string>Background</string>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stderr.log</string>
</dict>
</plist>
```

**Key differences from vllm.plist.j2:**

- No `HF_HOME` environment variable (llamacpp.service.j2 omits it -- line 9-10 vs vllm line 10)
- No `ExitTimeOut` (llamacpp.service.j2 has no `TimeoutStopSec` -- compare vllm line 34)
- `ProgramArguments` mirrors the llamacpp.service.j2 `ExecStart` logic:
  router mode (`--models-dir`) vs single-model mode (`--model`) branch at
  llamacpp.service.j2 lines 15-25
- Host defaults to `127.0.0.1` (llamacpp.service.j2 line 27), not `0.0.0.0`

### 3. fastapi.plist.j2

Reference: `systemd/fastapi.service.j2` (lines 1-27).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- {{ service.name }} -->
    <key>Label</key>
    <string>com.gpumod.{{ service.name }}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{{ settings.uvicorn_bin | default('uvicorn') }}</string>
        <string>{{ unit_vars.app_module | default('main:app') }}</string>
        <string>--port</string>
        <string>{{ service.port }}</string>
        <string>--host</string>
        <string>0.0.0.0</string>
{%- if unit_vars.workers is defined %}
        <string>--workers</string>
        <string>{{ unit_vars.workers }}</string>
{%- endif %}
{%- if unit_vars.extra_args is defined %}
{%-   for arg in unit_vars.extra_args.split() %}
        <string>{{ arg }}</string>
{%-   endfor %}
{%- endif %}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CUDA_VISIBLE_DEVICES</key>
        <string>{{ settings.cuda_devices | default('0') }}</string>
{%- for key, value in extra_env.items() %}
        <key>{{ key }}</key>
        <string>{{ value }}</string>
{%- endfor %}
    </dict>

    <key>WorkingDirectory</key>
    <string>{{ unit_vars.working_dir | default('/opt') }}</string>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ProcessType</key>
    <string>Background</string>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{{ settings.log_dir | default('~/Library/Logs/gpumod') }}/{{ service.name }}.stderr.log</string>
</dict>
</plist>
```

**Key differences from vllm/llamacpp:**

- Has `WorkingDirectory` (fastapi.service.j2 line 9)
- Simpler `ProgramArguments` (uvicorn with app_module, port, host, optional workers)
- No `HF_HOME` (FastAPI services don't download HF models)

### 4. mcp-server.plist.j2

Reference: `systemd/mcp-server.service.j2` (lines 1-20).
Rendered by `mcp_installer.py:52-102` with its own variable set.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- gpumod MCP Server -->
    <key>Label</key>
    <string>com.gpumod.mcp-server</string>

    <key>ProgramArguments</key>
    <array>
        <string>{{ python_bin }}</string>
        <string>-m</string>
        <string>gpumod.mcp_main</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{{ working_dir }}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{{ venv_bin }}:/usr/local/bin:/usr/bin:/bin</string>
        <key>GPUMOD_MCP_TRANSPORT</key>
        <string>{{ transport | default('streamable-http') }}</string>
        <key>GPUMOD_MCP_HOST</key>
        <string>{{ host | default('127.0.0.1') }}</string>
        <key>GPUMOD_MCP_PORT</key>
        <string>{{ port | default(8808) }}</string>
    </dict>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>5</integer>

    <key>ProcessType</key>
    <string>Background</string>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>{{ log_dir | default('~/Library/Logs/gpumod') }}/gpumod-mcp.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{{ log_dir | default('~/Library/Logs/gpumod') }}/gpumod-mcp.stderr.log</string>
</dict>
</plist>
```

**Key differences from service templates:**

- Uses `mcp_installer.py` variable namespace (`python_bin`, `venv_bin`,
  `working_dir`, `transport`, `host`, `port`) -- NOT the `service`/`settings`
  namespace (mcp_installer.py lines 90-101)
- Fixed label: `com.gpumod.mcp-server` (not variable-based)
- `ThrottleInterval=5` (matches `RestartSec=5` from mcp-server.service.j2 line 12,
  lower than ML services because MCP server starts fast)
- No `ExitTimeOut` (MCP server has no slow shutdown)

---

## SOLID Mapping

| Principle | Application |
|---|---|
| **Single Responsibility** | Each `.plist.j2` template owns one concern: generating plist XML for one driver type. Templates do not contain rendering logic or platform detection -- that belongs in `TemplateEngine` (`engine.py`). |
| **Open/Closed** | macOS support is added by creating **new** template files in a new `launchd/` directory. No existing systemd templates are modified. The `_DRIVER_TEMPLATE_MAP` extension is ticket gpumod-khc's scope. (`CLAUDE.md:80`, ADR lines 314-317) |
| **Liskov Substitution** | The rendered plist output is a string, same as the rendered systemd output. Any caller of `render_service_unit()` gets a string regardless of platform. The template contract (input variables, output type) is identical. |
| **Interface Segregation** | Templates consume only the variables they need. `mcp-server.plist.j2` uses the `mcp_installer.py` variable set; the other three use the `TemplateEngine.render_service_unit()` variable set. No template requires variables it does not use. |
| **Dependency Inversion** | Templates depend on abstract variable names (`service.name`, `service.port`, `settings.*`), not on concrete Python classes. The `TemplateEngine` injects the context dict -- templates are decoupled from the models module. (`engine.py:140-145`) |

---

## TDD Workflow

Test file: `tests/unit/templates/test_launchd_templates.py`

### RED Phase -- Write Failing Tests First

The tests follow the same pattern as `tests/unit/test_template_engine.py`
(existing systemd template tests, 809 lines). Key test categories:

**1. Valid plist XML output** -- Rendered output must parse with `plistlib.loads()`.

```python
import plistlib

class TestVllmPlistValid:
    def test_renders_valid_plist_xml(self, vllm_service, default_settings):
        engine = LaunchdTemplateEngine()  # or TemplateEngine with platform="darwin"
        result = engine.render_service_unit(vllm_service, default_settings)
        parsed = plistlib.loads(result.encode("utf-8"))
        assert isinstance(parsed, dict)
```

**2. Required keys present** -- Every plist must contain:
- `Label`
- `ProgramArguments` (array, non-empty)
- `KeepAlive` (dict with `SuccessfulExit=false`)
- `ThrottleInterval` (integer > 0)
- `ProcessType` = `"Background"`
- `RunAtLoad` = `false`
- `StandardOutPath`
- `StandardErrorPath`

```python
class TestVllmPlistRequiredKeys:
    def test_has_label(self, vllm_service, default_settings):
        result = render_vllm_plist(vllm_service, default_settings)
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["Label"] == f"com.gpumod.{vllm_service.name}"

    def test_has_keep_alive_successful_exit_false(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["KeepAlive"] == {"SuccessfulExit": False}

    def test_has_throttle_interval(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["ThrottleInterval"] == 30

    def test_has_process_type_background(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["ProcessType"] == "Background"

    def test_run_at_load_is_false(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["RunAtLoad"] is False
```

**3. Jinja2 variables rendered** -- Variables from the context are substituted.

```python
class TestVllmPlistVariables:
    def test_service_name_in_label(self, ...):
        assert "com.gpumod.vLLM Chat Service" in result

    def test_port_in_program_arguments(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        args = parsed["ProgramArguments"]
        port_idx = args.index("--port")
        assert args[port_idx + 1] == "8000"

    def test_model_id_in_program_arguments(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert "mistralai/Devstral-Small-2505" in parsed["ProgramArguments"]

    def test_extra_env_rendered(self, ...):
        extra_env = {"NCCL_P2P_DISABLE": "1"}
        result = render_vllm_plist(..., extra_env=extra_env)
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["EnvironmentVariables"]["NCCL_P2P_DISABLE"] == "1"
```

**4. No hardcoded paths** -- Templates must not contain `/home/`, `/Users/`,
or any absolute home directory path.

```python
class TestNoHardcodedPaths:
    def test_no_home_path_in_vllm(self, ...):
        result = render_vllm_plist(...)
        assert "/home/" not in result
        assert "/Users/" not in result

    def test_log_paths_use_variable(self, ...):
        result = render_vllm_plist(...)
        # Should use settings.log_dir or default ~/Library/Logs/gpumod
        assert "~/Library/Logs/gpumod" in result or "settings.log_dir" in result
```

**5. Conditional flags** -- Same conditional logic as systemd templates.

```python
class TestVllmPlistConditionals:
    def test_dtype_omitted_by_default(self, ...):
        parsed = plistlib.loads(result.encode("utf-8"))
        assert "--dtype" not in parsed["ProgramArguments"]

    def test_dtype_included_when_set(self, ...):
        unit_vars = {"dtype": "float16"}
        result = render_vllm_plist(..., unit_vars=unit_vars)
        parsed = plistlib.loads(result.encode("utf-8"))
        args = parsed["ProgramArguments"]
        assert "--dtype" in args
        assert "float16" in args

    def test_enforce_eager_flag(self, ...):
        unit_vars = {"enforce_eager": True}
        ...
        assert "--enforce-eager" in parsed["ProgramArguments"]
```

**6. Llamacpp router mode** -- Both `--model` and `--models-dir` branches.

```python
class TestLlamacppPlistRouterMode:
    def test_single_model_mode(self, ...):
        unit_vars = {"model_path": "/models/code.gguf"}
        ...
        assert "--model" in args
        assert "--models-dir" not in result

    def test_router_mode(self, ...):
        unit_vars = {"models_dir": "/opt/models", "no_models_autoload": True}
        ...
        assert "--models-dir" in args
        assert "--model" not in result
```

**7. MCP server template** -- Different variable namespace.

```python
class TestMcpServerPlist:
    def test_renders_valid_plist(self):
        result = render_mcp_plist(
            python_bin="/usr/bin/python3",
            venv_bin="/opt/gpumod/.venv/bin",
            working_dir="/opt/gpumod",
        )
        parsed = plistlib.loads(result.encode("utf-8"))
        assert parsed["Label"] == "com.gpumod.mcp-server"

    def test_program_arguments(self):
        ...
        assert parsed["ProgramArguments"] == [
            "/usr/bin/python3", "-m", "gpumod.mcp_main"
        ]

    def test_throttle_interval_is_5(self):
        # Lower than ML services (RestartSec=5 in mcp-server.service.j2:12)
        ...
        assert parsed["ThrottleInterval"] == 5
```

### GREEN Phase -- Create Templates

Write the four `.plist.j2` files with minimal content to pass all tests.

### REFACTOR Phase

- Extract shared plist boilerplate (XML declaration, DOCTYPE, KeepAlive block)
  into a Jinja2 macro or base template if repetition is excessive.
- Ensure consistent indentation (4 spaces, matching Apple plist conventions).

---

## Steps

1. **Create directory**: `src/gpumod/templates/launchd/` (empty).

2. **Write failing tests**: `tests/unit/templates/test_launchd_templates.py`.
   Use `plistlib.loads()` for XML validation. Mirror the fixture structure from
   `tests/unit/test_template_engine.py` (lines 16-61). Run `uv run pytest
   tests/unit/templates/test_launchd_templates.py` -- all tests must FAIL (RED).

3. **Create `vllm.plist.j2`**: Map every directive from
   `systemd/vllm.service.j2` to plist XML. Run tests -- vllm tests pass (GREEN).

4. **Create `llamacpp.plist.j2`**: Map from `systemd/llamacpp.service.j2`.
   Include router-mode branch. Run tests -- llamacpp tests pass (GREEN).

5. **Create `fastapi.plist.j2`**: Map from `systemd/fastapi.service.j2`.
   Include `WorkingDirectory`. Run tests -- fastapi tests pass (GREEN).

6. **Create `mcp-server.plist.j2`**: Map from `systemd/mcp-server.service.j2`.
   Use the `mcp_installer.py` variable namespace. Run tests -- mcp tests pass
   (GREEN).

7. **Run all quality gates**:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run mypy src/ --strict
   uv run pytest tests/unit/ -q
   ```
   All must pass with zero errors.

8. **REFACTOR**: If the four templates share significant boilerplate, consider
   a `_base.plist.j2` macro. Only if it reduces duplication without adding
   complexity.

---

## Acceptance Criteria

- [ ] Directory `src/gpumod/templates/launchd/` exists with four files:
      `vllm.plist.j2`, `llamacpp.plist.j2`, `fastapi.plist.j2`,
      `mcp-server.plist.j2`
- [ ] Each template renders valid plist XML parseable by `plistlib.loads()`
- [ ] Each plist contains required keys: `Label`, `ProgramArguments`,
      `KeepAlive` (with `SuccessfulExit=false`), `ThrottleInterval`,
      `ProcessType=Background`, `RunAtLoad=false`, `StandardOutPath`,
      `StandardErrorPath`
- [ ] `Label` follows `com.gpumod.{{ service.name }}` convention (or
      `com.gpumod.mcp-server` for the MCP template)
- [ ] `ProgramArguments` matches the `ExecStart` from the corresponding
      systemd template, split into array elements
- [ ] `EnvironmentVariables` dict matches `Environment=` directives from
      the corresponding systemd template
- [ ] All conditional flags (dtype, enforce_eager, runner, sleep_mode,
      flash_attn, jinja, router mode, workers, extra_args) render correctly
      in plist format
- [ ] Jinja2 variable names are identical to systemd templates:
      `service`, `settings`, `unit_vars`, `extra_env` for service templates;
      `python_bin`, `venv_bin`, `working_dir`, `transport`, `host`, `port`
      for mcp-server
- [ ] Log paths use `~/Library/Logs/gpumod/<name>.stdout.log` /
      `.stderr.log` pattern with no hardcoded home directory
- [ ] No `/home/`, `/Users/`, or machine-specific paths in templates
- [ ] `ThrottleInterval` is set in every template (30 for ML services,
      5 for MCP server)
- [ ] Tests exist in `tests/unit/templates/test_launchd_templates.py`
- [ ] All quality gates pass: lint, format, mypy, unit tests (zero errors)

---

## Edge Cases

1. **`extra_args` with spaces**: The systemd templates use `{{ unit_vars.extra_args }}`
   as a single string appended to `ExecStart`. In plist, each argument must be
   a separate `<string>` element. The template must split `extra_args` on
   whitespace: `{% for arg in unit_vars.extra_args.split() %}`. This means
   arguments with embedded spaces (e.g., paths with spaces) will break. Document
   this limitation. For paths with spaces, users should use `extra_args` as
   separate entries or quote-aware splitting can be added later.

2. **Empty `extra_env`**: When `extra_env` is `{}`, the `EnvironmentVariables`
   dict should still contain the base variables (CUDA_VISIBLE_DEVICES, etc.)
   but no extra entries. The `{% for %}` loop simply produces nothing.

3. **`service.name` with special characters**: plist `Label` values should be
   valid reverse-DNS identifiers. `service.name` may contain spaces or special
   characters (e.g., "vLLM Chat Service"). The `Label` field uses `service.name`
   directly. The `LaunchdController` (ticket gpumod-ong) is responsible for
   validation -- templates render what they receive. Tests should verify that
   names with spaces produce valid plist XML.

4. **`service.port` is None**: The `Service` model allows `port: int | None = None`
   (`models.py:71`). If `port` is `None`, the `--port` argument in
   `ProgramArguments` would render as `--port None`. Templates should NOT guard
   against this -- the caller (TemplateEngine or mcp_installer) must ensure port
   is set. This matches the systemd template behavior. But tests should
   document this assumption.

5. **Log directory does not exist**: `StandardOutPath` / `StandardErrorPath`
   specify file paths, but launchd does NOT create parent directories. The
   `ServiceFileInstaller` (ticket #8) or `LaunchdController` (ticket
   gpumod-ong) must `mkdir -p ~/Library/Logs/gpumod/` before loading the plist.
   Templates only specify the paths.

6. **MCP server without optional parameters**: When `transport`, `host`, or
   `port` are not passed, the template defaults take effect via Jinja2
   `| default()`. Verify that omitting all three still produces valid plist
   with the correct defaults (`streamable-http`, `127.0.0.1`, `8808`).

7. **`RunAtLoad=false` vs `true`**: The design choice is `false` because gpumod
   manages service lifecycle explicitly. If a user manually loads a plist via
   `launchctl load`, the service will NOT auto-start. This is intentional --
   users must use `gpumod start` or `launchctl bootstrap`. Document in template
   comments.

---

## Quality Gates

```bash
uv run ruff check src/ tests/           # Lint: 0 errors
uv run ruff format --check src/ tests/  # Format: no changes needed
uv run mypy src/ --strict               # Types: 0 errors
uv run pytest tests/unit/ -q            # Tests: all pass
```

All four must pass with zero errors before this ticket can be closed.

---

## Risks / Dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Depends on gpumod-khc** (template dispatch) for integration | Certain | Medium | Templates can be created and tested independently. They are plain Jinja2 files renderable by any `SandboxedEnvironment`. Integration with `TemplateEngine` happens when #6 lands. |
| **plist XML encoding edge cases** | Low | Low | Use `plistlib.loads()` in tests to validate every rendered template. plistlib is strict about XML well-formedness. |
| **`extra_args` splitting breaks quoted paths** | Medium | Low | Document limitation. Plist requires array arguments; there is no standard shell-quoting in plist XML. Can add `shlex.split()` in a follow-up if needed. |
| **Apple changes plist DTD** | Very Low | Low | The plist DTD (`PropertyList-1.0.dtd`) has been stable since macOS 10.0. |
| **`ExitTimeOut` key name** | Low | Medium | Apple documentation uses `ExitTimeOut` (not `ExitTimeout`). Verify spelling against `man launchd.plist`. If wrong, `launchctl` will silently ignore it. Test on a real macOS machine during integration. |
| **mcp_installer.py needs platform dispatch** | Certain | Low | mcp_installer.py currently hardcodes systemd paths (lines 19-21). It will need a platform branch to select `mcp-server.plist.j2`. This is part of ticket gpumod-khc scope or a small follow-up. The template itself can be created now. |
