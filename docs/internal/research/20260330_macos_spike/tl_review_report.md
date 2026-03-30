================================================================
SPIKE REVIEW: gpumod-2qc
================================================================

QUESTIONS:       16 total
  Answered:      15/16
  With evidence: 14/16
  Remaining:     1/16 (Q6: llama.cpp Metal perf — deferred, no local GGUF models)

Note: Q12-Q15 are architecture design questions answered in the ADR
(evidence = code analysis), not empirical measurements. Q4 memory
semantics used a toy model (stories260K, 1.1 MB) — directionally valid
but not a production-scale proof. Q16 Metal fatal patterns are
research-sourced, not verified against actual error output.

FINDINGS QUALITY:
  GPU metrics:   VERIFIED — Q1: 11 ioreg keys found, 3 required present.
                 Q2: Metal API returns 25,559 MB = 78%. Cross-confirmed by
                 llama.cpp init log (26800.60 MB). JSON matches findings doc.
                 Q3: pyobjc 12.1, pre-built wheel, 5 MB, MIT. Confirmed in
                 gpu_metrics_results.json. Q4: "Alloc system memory" is
                 monotonic with model load/unload; "In use" is volatile.
                 Methodology sound, toy model limits conclusion scope.

  vLLM compat:   VERIFIED — Q5: Both vllm-mlx (v0.2.6) and vllm-metal
                 (v0.1.0) pass 4/4 critical endpoints. 0/3 sleep/wake
                 (expected: unified memory). JSON results match findings
                 doc exactly. Startup times (3.2s vs 14.5s) documented.
                 Version mismatch (PyPI 0.2.6 vs __version__ 0.2.5) noted.

  launchd:       VERIFIED — Q7: bootstrap/bootout/kickstart/print all work.
                 Q8: ThrottleInterval=10 tolerates 10s startup. Q9:
                 KeepAlive.SuccessfulExit=false restarts exit(1) (runs=2)
                 but not exit(0) (runs=1). Results match QA report.

  Logging:       VERIFIED — Q10: log show latency 0.7-1.5s, file-based
                 fallback instant. Q16: Metal fatal patterns documented
                 (5 categories, 18 patterns). Patterns are research-sourced,
                 not empirically triggered — acceptable for spike scope.

  Architecture:  VERIFIED — ADR proposes ProcessController Protocol,
                 GPUMemoryTracker Protocol, template 2D dispatch, and
                 RAMInfoProvider Protocol. All align with existing code
                 structure. 3 options evaluated. 12 follow-up tickets
                 sequenced with dependency chain. See ADR Review below
                 for detailed assessment.

CODE REVIEW (scripts):
  Category A (Bugs):     3 findings
  Category B (Missing):  2 findings
  Category G (Security): 2 findings

  Details:

  A1. MAJOR — verify_vllm_endpoints.py:120
      TypeError crash when server is not running. `result.get('status_code',
      'ERR')` returns None (not 'ERR') because lines 80/84/88 explicitly set
      the key to None. The `:>3` format spec fails on NoneType.
      Fix: `result.get('status_code') or 'ERR'`

  A2. MINOR — verify_launchd.py:246
      In the finally block, if `plist_path` was never assigned (exception
      before line 127), `plist_path` is unbound and raises NameError.
      Extremely narrow race window. Acceptable for spike but note for
      reference.

  A3. MINOR — verify_launchd.py:113-114
      Uses `tempfile.mktemp()` (S306) for log paths. Creates a TOCTOU race.
      Low risk for spike scripts, but `NamedTemporaryFile(delete=False)` is
      the stdlib-recommended alternative.

  B1. MINOR — All 5 scripts: No top-level KeyboardInterrupt handler.
      verify_launchd.py is the only script with side effects (creates plists);
      Ctrl+C between Q7/Q8/Q9 could leave orphan plists. The per-question
      finally blocks mitigate most scenarios. Other 4 scripts: cosmetic only.

  B2. MINOR — verify_gpu_metrics.py: No Q4 function.
      Q4 (memory semantics under ML workload) was answered in the findings
      doc using a separate manual experiment with llama-server, not via this
      script. The script covers Q1-Q3 only. The findings doc documents the
      manual methodology, so the question IS answered — just not via the
      automated script. Acceptable for spike.

  G1. CRITICAL — .gitignore is broken.
      `docs/internal/research/20260330_macos_spike/.gitignore` contains the
      pattern `docs/internal/research/20260330_macos_spike/*_results.json`
      but .gitignore patterns are relative to the file's directory. The
      pattern never matches. `git check-ignore` exits 1 (not ignored).
      All 6 *_results.json files are committable, including launchd_results.json
      which contains PII.
      Fix: Change the pattern to `*_results.json`

  G2. CRITICAL — launchd_results.json contains PII.
      actual UID, home directory paths, and username appear in `command` fields
      and `output_preview` across 6 entries (lines 6, 13, 15, 21, 34, 42).
      The `systemctl_mapping` section correctly uses `<uid>` placeholders,
      proving the authors knew to sanitize — but missed the `tests[]` entries.
      Combined with G1 (.gitignore broken), this file WILL be committed.
      Fix: (a) Fix .gitignore. (b) Sanitize verify_launchd.py to replace
      `os.getuid()` with `<uid>` and `Path.home()` with `~` in serialized
      JSON.

ADR REVIEW:
  ProcessController:  ALIGNED
    The Protocol methods (start, stop, restart, is_active, get_state,
    get_logs, validate_service_name) map cleanly to both systemctl and
    launchctl. The existing ServiceDriver ABC (src/gpumod/services/base.py)
    operates at a higher level (HTTP health checks, sleep/wake) and does
    not touch the init system — ProcessController correctly separates
    this concern. Current direct imports of `systemd` in lifecycle.py:13,
    vllm.py:14, llamacpp.py:15, fastapi.py:10 confirm the coupling that
    the Protocol would remove. Sleep/wake exclusion from ProcessController
    is justified by Q5 empirical evidence (unified memory makes it a no-op).

  Template dispatch:  ALIGNED
    The 2D _DRIVER_TEMPLATE_MAP proposal extends the existing 1D map at
    engine.py:20. The _SAFE_NAME_RE at engine.py:27 already allows `.plist.j2`
    filenames without modification. The SandboxedEnvironment handles XML
    output. The UnitFileInstaller -> ServiceFileInstaller rename is
    consistent with the platform-neutral direction.

  DockerDriver scope: JUSTIFIED
    DockerDriver wraps `docker` CLI commands (no systemd dependency).
    The ADR correctly notes GPU passthrough is unavailable on Docker
    Desktop for macOS — this is a Docker Desktop limitation. Zero code
    changes is the right call.

  Evidence citations: VERIFIED
    - 78% (25,559 MB) matches gpu_metrics_results.json:
      `metal_recommended_max_mb: 25559`, `metal_actual_pct: 0.78`
    - 3.2s startup matches vllm_vllm-mlx_results.json (tested empirically)
    - 14.5s startup matches vllm_vllm-metal_results.json
    - vllm-mlx v0.2.6 and vllm-metal v0.1.0 versions match PyPI data
    - pyobjc 12.1, MIT, pre-built wheels — matches gpu_metrics_results.json
    - Sleep stub code references (worker.py:236-239, platform.py:159,
      v1/worker.py:355-369) are cited with line numbers — cannot verify
      without source, but internal consistency is strong
    - "In use system memory" volatility (469 -> 277 MB) in q2_q3_q4_findings.md
      is a plausible Metal behavior (compute buffers reclaimed post-inference)
    No fabricated results detected.

  Follow-up tickets:  COMPLETE
    12 tickets in correct dependency order. The sequence is:
    1. Protocol definitions (foundation)
    2-3. SystemdController + LaunchdController (parallel after 1)
    4-5. MetalGPUTracker + NvidiaGPUTracker (parallel after 1)
    6-7. Template dispatch + launchd templates (parallel after 1)
    8. ServiceFileInstaller (after 6-7)
    9. RAMInfoProvider (independent)
    10. LifecycleManager wiring (after 2-3)
    11. VRAMCheck wiring (after 4-5)
    12. Documentation (last)
    Sequencing is sound. No missing tickets identified for the proposed
    scope. One observation: there is no explicit ticket for platform
    detection / wiring the factory logic that selects SystemdController
    vs LaunchdController at startup. This is implicitly part of ticket 10
    but could be called out.

QA VALIDATION:   7 PASS, 3 FAIL across 10 tests
  Issue triage:
  - .gitignore: BLOCKING — prevents accidental PII commit. One-line fix.
  - PII in JSON: BLOCKING until .gitignore is fixed. If .gitignore is
    corrected, PII in launchd_results.json is ADVISORY (file won't be
    committed). Recommend sanitizing the script anyway for defense in depth.
  - vllm crash: ADVISORY — script already produced its results successfully
    (tested against running servers). The crash only occurs when probing
    a server that isn't running, which is an expected-failure path. The
    fix is trivial (one line) but does not block the spike.

PII CHECK:       FOUND
  - launchd_results.json: actual UID, home directory paths, username in
    `command` fields (6 occurrences) and `output_preview` (1 occurrence)
  - gpu_metrics_results.json: CLEAN (home path sanitized to ~)
  - vllm_vllm-mlx_results.json: CLEAN
  - vllm_vllm-metal_results.json: CLEAN
  - ram_portability_results.json: CLEAN
  - logging_results.json: CLEAN
  - All .py scripts: CLEAN (runtime-only os.getuid(), no hardcoded paths)
  - All .md findings docs: CLEAN

VERDICT: SPIKE COMPLETE — with 2 blocking items before merge

  Blocking items:
  1. Fix .gitignore: change pattern from
     `docs/internal/research/20260330_macos_spike/*_results.json`
     to `*_results.json`
  2. Sanitize launchd_results.json OR regenerate after fixing
     verify_launchd.py to replace UID/home paths with placeholders

  Advisory items:
  1. Fix verify_vllm_endpoints.py:120 crash
     (`result.get('status_code') or 'ERR'`)
  2. Add KeyboardInterrupt handler to verify_launchd.py main()
  3. Remove 3 unused imports (F401) across scripts
  4. Consider adding `# ruff: noqa: T201, S607, S603` headers to
     spike scripts if lint compliance is desired for the directory
  5. ADR follow-up ticket 10 should explicitly mention platform
     detection wiring (selecting controller at startup)

  Follow-up tickets needed (from ADR):
  1.  Protocol definitions (ProcessController, GPUMemoryTracker, RAMInfoProvider)
  2.  SystemdController (wrap existing systemd.py)
  3.  LaunchdController (launchctl bootstrap/bootout)
  4.  MetalGPUTracker (ioreg + pyobjc)
  5.  NvidiaGPUTracker (wrap existing VRAMTracker)
  6.  Template engine platform dispatch (2D map)
  7.  Launchd Jinja2 templates (vllm.plist.j2, llamacpp.plist.j2, fastapi.plist.j2)
  8.  ServiceFileInstaller (rename UnitFileInstaller, platform dispatch)
  9.  RAMInfoProvider (DarwinRAMInfo: sysctl + vm_stat)
  10. LifecycleManager wiring (inject ProcessController)
  11. VRAMCheck wiring (inject GPUMemoryTracker)
  12. Documentation update (architecture + CLI docs)

  Additional tickets (from review):
  13. Q6: llama.cpp Metal benchmark with real GGUF model (deferred from spike)
  14. Production-scale Q4 validation (memory release timing for 7B+ models)

================================================================
