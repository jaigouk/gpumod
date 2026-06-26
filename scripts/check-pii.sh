#!/usr/bin/env bash
# PII / privacy gate for this PUBLIC repository (github.com/jaigouk/gpumod).
#
# Everything committed here is published to a public GitHub remote, so personal
# information (real home-dir paths, usernames, machine brand/model) must never
# enter a commit. This gate scans the STAGED diff and blocks the commit if it
# finds any. It is invoked from .beads/hooks/pre-commit (the active hook, since
# core.hooksPath=.beads/hooks) and from scripts/pre-commit-check.sh.
#
# Incident that motivated it (2026-06-26): /home/<user> paths + machine brand
# leaked into .beads/issues.jsonl and reached the public remote. See CLAUDE.md
# "Privacy & Open Source".
#
# Escape hatch (rare, genuine false positive): SKIP_PII=1 git commit ...
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

if [ "${SKIP_PII:-0}" = "1" ]; then
    echo -e "  PII scan... ${YELLOW}SKIPPED${NC} (SKIP_PII=1)"
    exit 0
fi

# Added lines in the staged diff only (ignore the +++ file header lines).
added="$(git diff --cached --no-color -U0 | grep -E '^\+' | grep -vE '^\+\+\+' || true)"
if [ -z "$added" ]; then
    exit 0
fi

hits=""

# (1) Generic: real home-directory paths. Allow documented generic placeholders
#     (/home/user, /home/operator, /home/<user>, $HOME, ~).
home_hits="$(printf '%s\n' "$added" \
    | grep -nE '/home/[a-z][a-z0-9_-]*' \
    | grep -vE '/home/(user|operator|youruser)([^a-z0-9_-]|$)' || true)"
[ -n "$home_hits" ] && hits="${hits}\n${RED}[home-path]${NC}\n${home_hits}"

# (2) Project-specific forbidden strings (exact usernames / hostnames / real
#     name) live in a GITIGNORED .pii-blocklist so the sensitive strings are NOT
#     themselves committed into this gate. One extended-regex per line; blank
#     lines and #-comments ignored.
if [ -f .pii-blocklist ]; then
    patterns="$(grep -vE '^[[:space:]]*(#|$)' .pii-blocklist || true)"
    if [ -n "$patterns" ]; then
        block_hits="$(printf '%s\n' "$added" | grep -nEf <(printf '%s\n' "$patterns") || true)"
        [ -n "$block_hits" ] && hits="${hits}\n${RED}[blocklist]${NC}\n${block_hits}"
    fi
fi

if [ -n "$hits" ]; then
    echo -e "  PII scan... ${RED}FAILED${NC}"
    echo -e "Personal info found in staged changes (this is a PUBLIC repo):${hits}"
    echo ""
    echo "Scrub it: use ~, \$HOME, or generic placeholders (/home/<user>, the host)."
    echo "Genuine false positive? Re-run with: SKIP_PII=1 git commit ..."
    exit 1
fi

echo -e "  PII scan... ${GREEN}OK${NC}"
exit 0
