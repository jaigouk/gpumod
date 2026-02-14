#!/usr/bin/env python3
"""Bug Scenario: EASY - Off-by-one pagination error.

This bug is common in real codebases and relatively easy to spot.
The pagination skips the last page when items are evenly divisible by page_size.

To test:
    python easy_pagination.py          # Shows buggy behavior
    python easy_pagination.py --fixed  # Shows correct behavior
"""

from __future__ import annotations

import sys

# ============================================================================
# BUGGY CODE - presented to agents
# ============================================================================

BUGGY_CODE = '''
def paginate(items: list, page_size: int) -> list[list]:
    """Split items into pages of page_size."""
    pages = []
    total_pages = len(items) // page_size
    for page_num in range(total_pages):
        start = page_num * page_size
        end = start + page_size
        pages.append(items[start:end])
    return pages


def fetch_all_users(user_ids: list[int], batch_size: int = 10) -> list[dict]:
    """Fetch users in batches to avoid overwhelming the API."""
    results = []
    batches = paginate(user_ids, batch_size)
    for batch in batches:
        # Simulate API call
        for uid in batch:
            results.append({"id": uid, "name": f"User {uid}"})
    return results
'''


def paginate_buggy(items: list, page_size: int) -> list[list]:
    """Split items into pages of page_size."""
    pages = []
    total_pages = len(items) // page_size  # BUG: integer division loses remainder
    for page_num in range(total_pages):
        start = page_num * page_size
        end = start + page_size
        pages.append(items[start:end])
    return pages


def fetch_all_users_buggy(user_ids: list[int], batch_size: int = 10) -> list[dict]:
    """Fetch users in batches to avoid overwhelming the API."""
    results = []
    batches = paginate_buggy(user_ids, batch_size)
    for batch in batches:
        for uid in batch:
            results.append({"id": uid, "name": f"User {uid}"})
    return results


# ============================================================================
# FIXED CODE - expected solution
# ============================================================================

FIXED_CODE = '''
import math

def paginate(items: list, page_size: int) -> list[list]:
    """Split items into pages of page_size."""
    pages = []
    total_pages = math.ceil(len(items) / page_size)  # FIX: ceil to include partial page
    for page_num in range(total_pages):
        start = page_num * page_size
        end = start + page_size
        pages.append(items[start:end])
    return pages
'''

import math


def paginate_fixed(items: list, page_size: int) -> list[list]:
    """Split items into pages of page_size."""
    pages = []
    total_pages = math.ceil(len(items) / page_size)  # FIX: ceil to include partial page
    for page_num in range(total_pages):
        start = page_num * page_size
        end = start + page_size
        pages.append(items[start:end])
    return pages


def fetch_all_users_fixed(user_ids: list[int], batch_size: int = 10) -> list[dict]:
    """Fetch users in batches to avoid overwhelming the API."""
    results = []
    batches = paginate_fixed(user_ids, batch_size)
    for batch in batches:
        for uid in batch:
            results.append({"id": uid, "name": f"User {uid}"})
    return results


# ============================================================================
# TEST HARNESS
# ============================================================================


def test_pagination(use_fixed: bool = False) -> tuple[bool, str]:
    """Test pagination with various edge cases.

    Returns (passed, message) tuple.
    """
    paginate_fn = paginate_fixed if use_fixed else paginate_buggy
    fetch_fn = fetch_all_users_fixed if use_fixed else fetch_all_users_buggy

    errors = []

    # Test 1: Exact multiple of page_size (works in both)
    items = list(range(20))
    pages = paginate_fn(items, 10)
    if len(pages) != 2:
        errors.append(f"Test 1 FAIL: Expected 2 pages, got {len(pages)}")

    # Test 2: Non-multiple of page_size (BUG manifests here)
    items = list(range(25))
    pages = paginate_fn(items, 10)
    expected_pages = 3  # [0-9], [10-19], [20-24]
    if len(pages) != expected_pages:
        errors.append(f"Test 2 FAIL: Expected {expected_pages} pages, got {len(pages)}")

    # Test 3: Fetch all users - verify no data loss
    user_ids = list(range(1, 26))  # 25 users
    results = fetch_fn(user_ids, 10)
    if len(results) != 25:
        errors.append(
            f"Test 3 FAIL: Expected 25 users, got {len(results)} (lost {25 - len(results)})"
        )

    # Test 4: Single item
    items = [1]
    pages = paginate_fn(items, 10)
    if len(pages) != 1:
        errors.append(f"Test 4 FAIL: Expected 1 page, got {len(pages)}")

    # Test 5: Empty list
    pages = paginate_fn([], 10)
    if len(pages) != 0:
        errors.append(f"Test 5 FAIL: Expected 0 pages, got {len(pages)}")

    if errors:
        return False, "\n".join(errors)
    return True, "All tests passed!"


# ============================================================================
# SCORING HELPERS
# ============================================================================

BUG_KEYWORDS = [
    "off-by-one",
    "off by one",
    "integer division",
    "floor division",
    "remainder",
    "modulo",
    "last page",
    "partial page",
    "ceil",
    "math.ceil",
    "loses items",
    "missing items",
    "truncate",
]

FIX_KEYWORDS = [
    "math.ceil",
    "ceiling",
    "-(-len",  # Negative division trick
    "(len + page_size - 1)",  # Common integer ceiling pattern
    "// page_size + 1",  # Partial fix attempt
]


def score_response(response: str) -> dict:
    """Score an agent's response."""
    response_lower = response.lower()

    bug_found = any(kw in response_lower for kw in BUG_KEYWORDS)
    fix_proposed = any(kw in response_lower for kw in FIX_KEYWORDS)

    return {
        "bug_identified": bug_found,
        "fix_proposed": fix_proposed,
        "score": (2 if bug_found else 0) + (1 if fix_proposed else 0),
        "max_score": 3,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    use_fixed = "--fixed" in sys.argv

    mode = "FIXED" if use_fixed else "BUGGY"
    print(f"Running {mode} version...")
    print("-" * 40)

    passed, msg = test_pagination(use_fixed)
    print(msg)
    print("-" * 40)

    if passed:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)
