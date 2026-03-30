# QA Report: Spike gpumod-2qc Scripts

## Date: 2026-03-30

## PII AUDIT STATUS: 2 ISSUES FOUND

1. **BROKEN .gitignore** -- The local `.gitignore` uses the full path prefix
   `docs/internal/research/20260330_macos_spike/*_results.json` but this file lives
   inside the spike directory. Patterns in a `.gitignore` are relative to the file's
   location, so the pattern never matches. `git check-ignore` confirms the results
   files are NOT excluded. **Fix: change the pattern to `*_results.json`.**

2. **PII in launchd_results.json** -- The `command` fields and `output_preview`
   contain actual actual UID, home directory paths, and username embedded in paths.
   The `systemctl_mapping` section correctly uses `<uid>` placeholders, but the
   `tests[].command` fields contain raw values. This is acceptable only if the
   `.gitignore` is fixed (Issue 1 above). The script should sanitize
   `command` strings before writing JSON.

No PII was found in:
- `gpu_metrics_results.json` (home path sanitized to `~`)
- `logging_results.json` (clean)
- `ram_portability_results.json` (clean)
- `vllm_vllm-mlx_results.json` (clean)
- `vllm_vllm-metal_results.json` (clean)
- `q2_q3_q4_findings.md` (clean)
- `q5_q6_findings.md` (clean)
- `README.md` (clean)
- All `.py` scripts (no hardcoded paths; `os.getuid()` is runtime-only)

---

## Validation Results

| # | Test | Result | Evidence |
|---|------|--------|----------|
| 1 | GPU metrics script | **PASS** | All 3 required ioreg keys present (`Alloc system memory`, `In use system memory`, `Device Utilization %`). 11 total keys found. No crashes. Metal API returned 25,559 MB (78% of 32 GB). |
| 2 | launchd verification | **PASS** | Q7: bootstrap/bootout/kickstart/print all work (returncode=0). Q8: launchd tolerates 10s startup with ThrottleInterval=10. Q9: KeepAlive.SuccessfulExit=false correctly restarts on exit(1) (runs=2) but not exit(0) (runs=1). Clean cleanup. |
| 3 | logging verification | **PASS** | `log show` by process: 1.49s latency, found test message. `log show` by message content: 0.70s latency, found. Subsystem query works (empty as expected). File-based fallback: instant. Average latency 1.09s. |
| 4 | RAM portability script | **PASS** | `sysctl hw.memsize`: 32,768 MB. `vm_stat` estimated available: 14,508 MB. `os.sysconf` total: 32,768 MB (no `SC_AVPHYS_PAGES` on macOS). `psutil` not installed (correctly handled with ImportError). |
| 5 | PII check on result JSON | **FAIL** | `launchd_results.json` contains actual UID, home paths, and venv paths in command strings. Other 4 result files are clean. |
| 6 | PII check on Python scripts | **PASS** | No hardcoded paths found. `os.getuid()` in `verify_launchd.py` is runtime-necessary for `launchctl gui/<uid>` syntax. |
| 7 | Ruff lint check | **FAIL** | 206 errors: 131 T201 (print), 24 S607 (partial path), 17 E501 (line too long), 16 S603 (subprocess), 6 S306 (mktemp), 3 F401 (unused import), 2 F541 (f-string), 1 E741, 1 F841, 1 PLR0915, 1 S103, 1 SIM105, 1 SIM108, 1 SIM115. See analysis below. |
| 8 | .gitignore check | **FAIL** | Pattern `docs/internal/research/20260330_macos_spike/*_results.json` in local `.gitignore` does not match (wrong relative path). `git check-ignore` returns exit 1. Root `.gitignore` has no results pattern either. **Results files are committable.** |
| 9 | Launchd cleanup verification | **PASS** | No orphan plists found in `~/Library/LaunchAgents/com.gpumod.spike*` before or after running `verify_launchd.py`. |
| 10 | Cross-check ioreg keys | **PASS** | All 8 keys documented in `macos-gpu-memory-launchd.md` are present in live `ioreg` output. 3 additional undocumented keys found (`Allocated PB Size`, `SplitSceneCount`, `TiledSceneBytes`). The script correctly checks only the 3 keys gpumod needs. |

---

## Edge Cases

| Case | Description | Result | Notes |
|------|-------------|--------|-------|
| A | pyobjc not installed | **PASS** | `verify_gpu_metrics.py` line 123-152: `try: import Metal ... except ImportError:` prints install instructions and continues. Q2 budget calculation falls back to sysctl-only. |
| B | verify_launchd.py interruption | **PASS (with caveat)** | Each test function (Q7/Q8/Q9) has a `finally` block that calls `_unload()` and cleans up plist, script, and log files. Verified: no orphan plists after full run. **Caveat:** If KeyboardInterrupt fires before `plist_path` is assigned (line 127), the `finally` block at line 246 would raise `NameError` because `plist_path` is referenced in the conditional. This is an extremely narrow race window and low risk for a spike script. |
| C | verify_vllm_endpoints.py with no server | **FAIL** | **Bug found.** Script crashes with `TypeError: unsupported format string passed to NoneType.__format__` at line 120. Root cause: `result.get('status_code', 'ERR')` returns `None` (not `'ERR'`) because the key exists with value `None`. The format string `{...:>3}` fails on `NoneType`. Fix: use `result.get('status_code') or 'ERR'` instead. |
| D | KeyboardInterrupt handling | **PARTIAL** | Only the helper script inside `_create_slow_server_script()` handles `KeyboardInterrupt`. None of the 5 main scripts wrap their `main()` in try/except for KeyboardInterrupt. For `verify_launchd.py` this means Ctrl+C during inter-test `time.sleep()` could skip cleanup of the currently active test. The `finally` blocks within each Q function mitigate this for in-progress tests, but if interrupted between Q7 and Q8, only Q7 cleanup runs. For the other 4 scripts (no side effects), this is not a concern. |

---

## Ruff Lint Analysis

**Total: 206 errors across 5 scripts.**

These are spike/research scripts, not production code. The errors fall into expected categories:

| Rule | Count | Verdict | Rationale |
|------|-------|---------|-----------|
| T201 (print) | 131 | **Expected** | Spike scripts use `print()` for console output by design. |
| S607 (partial path) | 24 | **Expected** | `ioreg`, `sysctl`, `launchctl`, `logger`, `vm_stat` are system binaries. Full paths would reduce portability. |
| E501 (line too long) | 17 | **Minor** | Long format strings with inline calculations. |
| S603 (subprocess) | 16 | **Expected** | Spike scripts call system tools via subprocess. No user input. |
| S306 (mktemp) | 6 | **Low risk** | `tempfile.mktemp()` used in `verify_launchd.py` for log paths. Race condition is theoretical for spike scripts but `NamedTemporaryFile` would be safer. |
| F401 (unused import) | 3 | **Fix recommended** | Unused imports waste reader attention. |
| F541 (f-string no placeholders) | 2 | **Fix recommended** | e.g., `f"string"` should be `"string"`. |
| Other (E741, F841, PLR0915, S103, SIM*) | 5 | **Low priority** | Minor style issues in spike code. |

**Recommendation:** Fix F401 and F541 (trivial). Accept T201/S607/S603/E501 as inherent to spike scripts. Consider adding a `# ruff: noqa` comment at the top of each spike script if lint compliance is required for the directory.

---

## Issues Found

### Issue 1: BROKEN .gitignore (Severity: High)

- **File:** `docs/internal/research/20260330_macos_spike/.gitignore`
- **Root Cause:** The pattern uses the full path from repo root (`docs/internal/research/20260330_macos_spike/*_results.json`) but `.gitignore` patterns are relative to the `.gitignore` file's directory. The pattern never matches.
- **Impact:** All 5 `*_results.json` files show as untracked in `git status` and could be accidentally committed, leaking PII (UIDs, home paths).
- **Fix:** Change the pattern to `*_results.json`.

### Issue 2: PII in launchd_results.json (Severity: Medium)

- **File:** `docs/internal/research/20260330_macos_spike/launchd_results.json`
- **Root Cause:** `verify_launchd.py` serializes raw `launchctl` command strings containing `gui/{uid}` with the actual UID and full plist paths including the home directory. The `systemctl_mapping` dict correctly uses `<uid>` placeholders, but `tests[].command` does not.
- **Impact:** If `.gitignore` is fixed (Issue 1), this file will not be committed. If accidentally committed, it leaks UID and home directory.
- **Fix:** Sanitize command strings before writing to JSON by replacing `os.getuid()` with `<uid>` and `Path.home()` with `~` in all serialized data.

### Issue 3: verify_vllm_endpoints.py crash on connection refused (Severity: Medium)

- **File:** `docs/internal/research/20260330_macos_spike/verify_vllm_endpoints.py`, line 120
- **Root Cause:** `result.get('status_code', 'ERR')` returns `None` (not `'ERR'`) when the key exists but was explicitly set to `None` at line 80/84. Python's `dict.get(key, default)` only returns the default when the key is absent, not when it maps to `None`. The `None` value then fails with `:>3` format specification.
- **Impact:** The script exits with a traceback instead of reporting "connection refused" gracefully.
- **Fix:** Replace `result.get('status_code', 'ERR')` with `result.get('status_code') or 'ERR'` at line 120.

### Issue 4: No KeyboardInterrupt handling in main() functions (Severity: Low)

- **Files:** All 5 `verify_*.py` scripts
- **Root Cause:** No top-level `try/except KeyboardInterrupt` around `main()`.
- **Impact:** For `verify_launchd.py`, Ctrl+C could leave orphan plists if the interrupt occurs at specific moments. For other scripts, the only impact is an ugly traceback. The `finally` blocks in `verify_launchd.py` mitigate most scenarios.
- **Fix:** Wrap `main()` call in `except KeyboardInterrupt: print("\nInterrupted"); sys.exit(130)`. For `verify_launchd.py`, add an `atexit` handler that scans for `com.gpumod.spike*` plists.

---

## Recommendations

### Must Fix (before merging spike branch)

1. **Fix `.gitignore`** -- Change the pattern from `docs/internal/research/20260330_macos_spike/*_results.json` to `*_results.json`. This is a one-line fix that prevents accidental PII commits.

### Should Fix

2. **Fix vllm_endpoints crash** -- Replace `result.get('status_code', 'ERR')` with `result.get('status_code') or 'ERR'` on line 120. Alternatively, set `result["status_code"] = "ERR"` (string) instead of `None` in the except blocks (lines 80, 84, 89).

3. **Sanitize launchd_results.json** -- In `verify_launchd.py`, replace the actual UID and home path in serialized command strings with `<uid>` and `~` before writing JSON. Model this on the existing sanitization in `gpu_metrics_results.py` line 190.

### Nice to Have

4. **Fix unused imports** -- Remove 3 unused imports (F401) and 2 placeholder-less f-strings (F541).

5. **Add KeyboardInterrupt handling** -- Wrap `if __name__ == "__main__": main()` blocks in all scripts with `except KeyboardInterrupt`.

6. **Add `# ruff: noqa: T201, S607, S603` to spike scripts** -- Explicitly mark these as acceptable for research scripts, keeping the lint summary clean.

---

## Test Environment

| Property | Value |
|----------|-------|
| Machine | Apple M2 Max, 30 GPU cores |
| RAM | 32 GB unified memory |
| macOS | 26.4 (Darwin 25.4.0) |
| Python | 3.12.7 (arm64) |
| pyobjc-framework-Metal | 12.1 |
| llama-server | build 3614 (Homebrew) |
| Branch | macos |
