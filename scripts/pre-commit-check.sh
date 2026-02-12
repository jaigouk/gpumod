#!/usr/bin/env bash
# Pre-commit quality gates for gpumod
# Runs: ruff check, ruff format --check, mypy, pytest (optional)
#
# Exit codes:
#   0 - All checks passed
#   1 - Quality check failed
#
# Environment variables:
#   SKIP_TESTS=1     - Skip pytest (faster commits during iteration)
#   SKIP_PRECOMMIT=1 - Skip all checks (emergency escape hatch)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Emergency escape hatch
if [ "${SKIP_PRECOMMIT:-0}" = "1" ]; then
    echo -e "${YELLOW}SKIP_PRECOMMIT=1: Skipping all quality checks${NC}"
    exit 0
fi

echo "Running pre-commit quality checks..."

# 1. Ruff lint check
echo -n "  Ruff lint... "
if uv run ruff check src/ tests/ --quiet 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo "Fix with: uv run ruff check src/ tests/ --fix"
    exit 1
fi

# 2. Ruff format check
echo -n "  Ruff format... "
if uv run ruff format --check src/ tests/ --quiet 2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo "Fix with: uv run ruff format src/ tests/"
    exit 1
fi

# 3. Mypy type check
echo -n "  Mypy types... "
if uv run mypy src/ --strict 2>/dev/null | grep -q "Success"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo "Run: uv run mypy src/ --strict"
    exit 1
fi

# 4. Pytest (optional, can be slow)
if [ "${SKIP_TESTS:-0}" = "1" ]; then
    echo -e "  Pytest... ${YELLOW}SKIPPED${NC} (SKIP_TESTS=1)"
else
    echo -n "  Pytest... "
    if uv run pytest tests/unit -q --tb=no -x 2>/dev/null | tail -1 | grep -qE "passed|no tests"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo ""
        echo "Run: uv run pytest tests/unit -v"
        exit 1
    fi
fi

echo -e "${GREEN}All quality checks passed!${NC}"
exit 0
