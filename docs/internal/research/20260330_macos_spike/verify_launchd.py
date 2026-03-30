"""Spike Q7-Q9: Verify launchd lifecycle, ThrottleInterval, and KeepAlive behavior.

Run on macOS to answer:
  Q7. launchctl lifecycle semantics (load vs bootstrap, unload vs bootout)
  Q8. ThrottleInterval vs slow-starting ML services (30s+)
  Q9. KeepAlive.SuccessfulExit — SIGTERM vs crash distinction

Usage:
    uv run python docs/internal/research/20260330_macos_spike/verify_launchd.py

Requires: macOS. Creates temporary plists in ~/Library/LaunchAgents/.
Cleans up after itself.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LABEL_PREFIX = "com.gpumod.spike"


def check_platform() -> bool:
    if sys.platform != "darwin":
        print("SKIP: Not macOS")
        return False
    return True


def _write_plist(label: str, plist: dict) -> Path:
    """Write a plist to ~/Library/LaunchAgents/ and return the path."""
    path = LAUNCH_AGENTS / f"{label}.plist"
    with path.open("wb") as f:
        plistlib.dump(plist, f)
    return path


def _unload(label: str, plist_path: Path) -> None:
    """Best-effort unload and cleanup."""
    uid = os.getuid()
    # Try modern bootout first, fallback to legacy unload
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{label}"],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
    )
    if plist_path.exists():
        plist_path.unlink()


def _is_loaded(label: str) -> bool:
    """Check if a launchd job is loaded."""
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{label}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _create_slow_server_script() -> Path:
    """Create a Python script that takes N seconds to start (simulates model loading)."""
    script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="gpumod_spike_slow_", delete=False
    )
    script.write("""\
import http.server
import os
import sys
import time

startup_delay = int(os.environ.get("STARTUP_DELAY", "5"))
port = int(os.environ.get("PORT", "18999"))
exit_code = int(os.environ.get("EXIT_CODE", "0"))

# If EXIT_CODE is set and non-zero, exit immediately (for Q9 testing)
if os.environ.get("IMMEDIATE_EXIT") == "1":
    sys.exit(exit_code)

print(f"Starting with {startup_delay}s delay on port {port}...", flush=True)
time.sleep(startup_delay)
print(f"Ready! Serving on port {port}", flush=True)

server = http.server.HTTPServer(("127.0.0.1", port), http.server.SimpleHTTPRequestHandler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
""")
    script.close()
    os.chmod(script.name, 0o755)
    return Path(script.name)


def q7_lifecycle_semantics() -> dict:
    """Q7: Test launchctl load/bootstrap/unload/bootout/print/kickstart."""
    print("\n=== Q7: launchctl lifecycle semantics ===")
    findings: dict = {"tests": []}
    uid = os.getuid()
    label = f"{LABEL_PREFIX}.q7-lifecycle"
    script = _create_slow_server_script()
    log_out = Path(tempfile.mktemp(suffix=".stdout.log"))
    log_err = Path(tempfile.mktemp(suffix=".stderr.log"))

    plist = {
        "Label": label,
        "ProgramArguments": [sys.executable, str(script)],
        "EnvironmentVariables": {"STARTUP_DELAY": "2", "PORT": "18990"},
        "StandardOutPath": str(log_out),
        "StandardErrorPath": str(log_err),
        "RunAtLoad": True,
        "KeepAlive": False,
    }

    try:
        plist_path = _write_plist(label, plist)

        # Test 1: launchctl bootstrap (modern API)
        print(f"  Test 1: launchctl bootstrap gui/{uid} {plist_path}")
        r = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "bootstrap",
            "command": f"launchctl bootstrap gui/{uid} {plist_path}",
            "returncode": r.returncode,
            "stderr": r.stderr.strip(),
            "loaded": _is_loaded(label),
        })
        print(f"    returncode={r.returncode}, loaded={_is_loaded(label)}")
        if r.stderr.strip():
            print(f"    stderr: {r.stderr.strip()}")
        time.sleep(3)

        # Test 2: launchctl print (status equivalent)
        print(f"  Test 2: launchctl print gui/{uid}/{label}")
        r = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "print",
            "command": f"launchctl print gui/{uid}/{label}",
            "returncode": r.returncode,
            "output_preview": r.stdout[:500] if r.stdout else "",
            "has_pid": "pid =" in r.stdout.lower() if r.stdout else False,
            "has_state": "state =" in r.stdout.lower() if r.stdout else False,
        })
        print(f"    returncode={r.returncode}")
        if r.stdout:
            # Extract key info
            for line in r.stdout.splitlines():
                line = line.strip()
                if any(k in line.lower() for k in ["pid =", "state =", "last exit"]):
                    print(f"    {line}")

        # Test 3: launchctl kickstart -k (restart equivalent)
        print(f"  Test 3: launchctl kickstart -kp gui/{uid}/{label}")
        r = subprocess.run(
            ["launchctl", "kickstart", "-kp", f"gui/{uid}/{label}"],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "kickstart",
            "command": f"launchctl kickstart -kp gui/{uid}/{label}",
            "returncode": r.returncode,
            "stderr": r.stderr.strip(),
        })
        print(f"    returncode={r.returncode}")
        time.sleep(3)

        # Test 4: launchctl bootout (modern unload)
        print(f"  Test 4: launchctl bootout gui/{uid}/{label}")
        r = subprocess.run(
            ["launchctl", "bootout", f"gui/{uid}/{label}"],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "bootout",
            "command": f"launchctl bootout gui/{uid}/{label}",
            "returncode": r.returncode,
            "stderr": r.stderr.strip(),
            "still_loaded": _is_loaded(label),
        })
        print(f"    returncode={r.returncode}, still_loaded={_is_loaded(label)}")

        # Test 5: legacy launchctl load (for comparison)
        print(f"  Test 5: launchctl load {plist_path} (legacy)")
        r = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "load_legacy",
            "command": f"launchctl load {plist_path}",
            "returncode": r.returncode,
            "stderr": r.stderr.strip(),
            "deprecated_warning": "deprecated" in r.stderr.lower() if r.stderr else False,
            "loaded": _is_loaded(label),
        })
        print(f"    returncode={r.returncode}, deprecated_warning={'deprecated' in r.stderr.lower() if r.stderr else False}")
        time.sleep(2)

        # Test 6: legacy launchctl unload
        print(f"  Test 6: launchctl unload {plist_path} (legacy)")
        r = subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True, text=True,
        )
        findings["tests"].append({
            "name": "unload_legacy",
            "command": f"launchctl unload {plist_path}",
            "returncode": r.returncode,
            "stderr": r.stderr.strip(),
        })
        print(f"    returncode={r.returncode}")

        # Summary: map to systemctl equivalents (use <uid> placeholder, not actual UID)
        findings["systemctl_mapping"] = {
            "start": "launchctl bootstrap gui/<uid> <plist> OR launchctl kickstart gui/<uid>/<label>",
            "stop": "launchctl bootout gui/<uid>/<label>",
            "restart": "launchctl kickstart -kp gui/<uid>/<label>",
            "is-active": "launchctl print gui/<uid>/<label> (check exit code + 'state' field)",
            "status": "launchctl print gui/<uid>/<label> (full output)",
            "enable": "RunAtLoad=true in plist (no runtime equivalent)",
            "disable": "RunAtLoad=false in plist (no runtime equivalent)",
            "show": "launchctl print gui/<uid>/<label>",
        }
        print("\n  systemctl → launchctl mapping:")
        for k, v in findings["systemctl_mapping"].items():
            print(f"    systemctl --user {k} → {v}")

    finally:
        _unload(label, _write_plist(label, plist) if not plist_path.exists() else plist_path)
        script.unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        log_err.unlink(missing_ok=True)
        plist_path = LAUNCH_AGENTS / f"{label}.plist"
        plist_path.unlink(missing_ok=True)

    return findings


def q8_throttle_interval() -> dict:
    """Q8: Test ThrottleInterval with slow-starting process (simulated model load)."""
    print("\n=== Q8: ThrottleInterval vs slow startup ===")
    findings: dict = {"tests": []}
    uid = os.getuid()
    label = f"{LABEL_PREFIX}.q8-throttle"
    script = _create_slow_server_script()
    log_out = Path(tempfile.mktemp(suffix=".stdout.log"))
    log_err = Path(tempfile.mktemp(suffix=".stderr.log"))

    # Test: KeepAlive=true with 45s startup delay and ThrottleInterval=10
    plist = {
        "Label": label,
        "ProgramArguments": [sys.executable, str(script)],
        "EnvironmentVariables": {"STARTUP_DELAY": "10", "PORT": "18991"},
        "StandardOutPath": str(log_out),
        "StandardErrorPath": str(log_err),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
    }
    plist_path = _write_plist(label, plist)

    try:
        print(f"  Loading with STARTUP_DELAY=10s, ThrottleInterval=10...")
        r = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
            capture_output=True, text=True,
        )
        print(f"  bootstrap: returncode={r.returncode}")

        # Check at 5s (should still be starting)
        time.sleep(5)
        loaded_5s = _is_loaded(label)
        print(f"  At 5s: loaded={loaded_5s}")

        # Check at 12s (should be running now)
        time.sleep(7)
        loaded_12s = _is_loaded(label)
        r = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True, text=True,
        )
        state_12s = ""
        for line in (r.stdout or "").splitlines():
            if "state" in line.lower() or "pid" in line.lower():
                state_12s += line.strip() + "; "
        print(f"  At 12s: loaded={loaded_12s}, state={state_12s}")

        findings["tests"].append({
            "name": "slow_startup_10s_delay",
            "throttle_interval": 10,
            "startup_delay": 10,
            "loaded_at_5s": loaded_5s,
            "loaded_at_12s": loaded_12s,
            "state_at_12s": state_12s,
            "conclusion": "launchd did NOT kill the slow-starting process" if loaded_12s else "launchd KILLED the slow-starting process",
        })
        print(f"  Conclusion: {'OK — launchd tolerates slow startup' if loaded_12s else 'PROBLEM — launchd killed slow process'}")

    finally:
        _unload(label, plist_path)
        script.unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        log_err.unlink(missing_ok=True)
        plist_path.unlink(missing_ok=True)

    return findings


def q9_keepalive_behavior() -> dict:
    """Q9: Test KeepAlive.SuccessfulExit behavior."""
    print("\n=== Q9: KeepAlive.SuccessfulExit behavior ===")
    findings: dict = {"tests": []}
    uid = os.getuid()
    label = f"{LABEL_PREFIX}.q9-keepalive"
    script = _create_slow_server_script()
    log_out = Path(tempfile.mktemp(suffix=".stdout.log"))

    scenarios = [
        ("exit_0", {"IMMEDIATE_EXIT": "1", "EXIT_CODE": "0"}, "Process exits 0 — should NOT restart"),
        ("exit_1", {"IMMEDIATE_EXIT": "1", "EXIT_CODE": "1"}, "Process exits 1 — should restart"),
    ]

    for scenario_name, env_vars, description in scenarios:
        print(f"\n  Scenario: {description}")
        test_label = f"{label}-{scenario_name}"

        plist = {
            "Label": test_label,
            "ProgramArguments": [sys.executable, str(script)],
            "EnvironmentVariables": {**env_vars, "PORT": "18992"},
            "StandardOutPath": str(log_out),
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 5,
        }
        plist_path = _write_plist(test_label, plist)

        try:
            r = subprocess.run(
                ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
                capture_output=True, text=True,
            )
            time.sleep(8)  # Wait for potential restart cycle

            r = subprocess.run(
                ["launchctl", "print", f"gui/{uid}/{test_label}"],
                capture_output=True, text=True,
            )

            # Count spawn attempts from print output
            spawn_count = ""
            last_exit = ""
            for line in (r.stdout or "").splitlines():
                l = line.strip().lower()
                if "spawn count" in l or "runs" in l:
                    spawn_count = line.strip()
                if "last exit" in l:
                    last_exit = line.strip()

            loaded = r.returncode == 0
            findings["tests"].append({
                "name": scenario_name,
                "description": description,
                "env": env_vars,
                "still_loaded": loaded,
                "spawn_count": spawn_count,
                "last_exit": last_exit,
            })
            print(f"    still_loaded={loaded}, {spawn_count}, {last_exit}")

        finally:
            _unload(test_label, plist_path)
            plist_path = LAUNCH_AGENTS / f"{test_label}.plist"
            plist_path.unlink(missing_ok=True)

    # Cleanup
    script.unlink(missing_ok=True)
    log_out.unlink(missing_ok=True)
    return findings


def _sanitize_findings(obj: object) -> object:
    """Recursively replace home paths and UIDs with placeholders."""
    home = str(Path.home())
    uid = str(os.getuid())
    if isinstance(obj, str):
        return obj.replace(home, "~").replace(f"gui/{uid}", "gui/<uid>")
    if isinstance(obj, dict):
        return {k: _sanitize_findings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_findings(v) for v in obj]
    return obj


def main() -> None:
    if not check_platform():
        sys.exit(0)

    all_findings: dict = {}
    all_findings["q7_lifecycle"] = q7_lifecycle_semantics()
    all_findings["q8_throttle"] = q8_throttle_interval()
    all_findings["q9_keepalive"] = q9_keepalive_behavior()

    # Sanitize PII before writing
    sanitized = _sanitize_findings(all_findings)

    out_path = "docs/internal/research/20260330_macos_spike/launchd_results.json"
    with open(out_path, "w") as f:
        json.dump(sanitized, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
