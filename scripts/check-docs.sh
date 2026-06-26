#!/usr/bin/env bash
# Docs build gate: catch `mkdocs build --strict` failures (broken cross-tree
# links, bad nav, unresolved refs) BEFORE commit — the GitHub "Deploy docs"
# workflow runs the same --strict build, so a green local check keeps CI green.
#
# Only runs when docs/ or mkdocs.yml is staged (non-docs commits stay fast).
# Invoked from .beads/hooks/pre-commit (the active hook) and from
# scripts/pre-commit-check.sh. Escape: SKIP_DOCS=1 git commit ...
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

if [ "${SKIP_DOCS:-0}" = "1" ]; then
    echo -e "  Docs build... ${YELLOW}SKIPPED${NC} (SKIP_DOCS=1)"
    exit 0
fi

# Skip unless this commit touches docs or the mkdocs config.
staged="$(git diff --cached --name-only)"
if ! printf '%s\n' "$staged" | grep -qE '^(docs/|mkdocs\.ya?ml$)'; then
    exit 0
fi

# Skip gracefully if the (optional) docs toolchain isn't installed — CI is the
# authoritative gate; the pre-commit check is a fast local catch.
if ! uv run --no-sync mkdocs --version >/dev/null 2>&1; then
    echo -e "  Docs build... ${YELLOW}SKIPPED${NC} (mkdocs not installed; run 'uv sync --group docs')"
    exit 0
fi

echo -n "  Docs (mkdocs --strict)... "
site_dir="$(mktemp -d)"
trap 'rm -rf "$site_dir"' EXIT
if uv run --no-sync mkdocs build --strict --site-dir "$site_dir" >/tmp/mkdocs-precommit.log 2>&1; then
    echo -e "${GREEN}OK${NC}"
    exit 0
else
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo "mkdocs --strict found problems (broken links / nav). Details:"
    grep -E "WARNING|ERROR|Aborted" /tmp/mkdocs-precommit.log | head -20
    echo ""
    echo "Reproduce: uv run mkdocs build --strict"
    echo "Genuine false positive? SKIP_DOCS=1 git commit ..."
    exit 1
fi
