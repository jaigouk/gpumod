#!/usr/bin/env python3
"""Bug Scenario: MEDIUM - Resource leak in connection pool.

The connection is acquired but not properly released on exception.
This is a common bug pattern in database/network code.

To test:
    python medium_resource_leak.py          # Shows buggy behavior (connection leak)
    python medium_resource_leak.py --fixed  # Shows correct behavior
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

# ============================================================================
# BUGGY CODE - presented to agents
# ============================================================================

BUGGY_CODE = '''
class ConnectionPool:
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.available: list[Connection] = []
        self.in_use: set[Connection] = set()

        # Pre-create connections
        for i in range(max_connections):
            self.available.append(Connection(id=i))

    def acquire(self) -> Connection:
        """Get a connection from the pool."""
        if not self.available:
            raise RuntimeError("No connections available")
        conn = self.available.pop()
        self.in_use.add(conn)
        return conn

    def release(self, conn: Connection) -> None:
        """Return a connection to the pool."""
        if conn in self.in_use:
            self.in_use.remove(conn)
            self.available.append(conn)


class DatabaseClient:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def execute_query(self, query: str) -> list[dict]:
        """Execute a query and return results."""
        conn = self.pool.acquire()
        result = conn.execute(query)
        self.pool.release(conn)
        return result

    def execute_batch(self, queries: list[str]) -> list[list[dict]]:
        """Execute multiple queries."""
        results = []
        for query in queries:
            conn = self.pool.acquire()
            result = conn.execute(query)
            results.append(result)
            self.pool.release(conn)
        return results
'''


@dataclass(unsafe_hash=True)
class Connection:
    id: int
    _closed: bool = field(default=False, compare=False, hash=False)

    def execute(self, query: str) -> list[dict]:
        """Simulate query execution."""
        if self._closed:
            raise RuntimeError("Connection is closed")
        # Simulate failure for specific queries
        if "FAIL" in query:
            raise ValueError(f"Query failed: {query}")
        return [{"result": f"data from conn {self.id}"}]

    def close(self) -> None:
        self._closed = True


class ConnectionPoolBuggy:
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.available: list[Connection] = []
        self.in_use: set[Connection] = set()

        for i in range(max_connections):
            self.available.append(Connection(id=i))

    def acquire(self) -> Connection:
        if not self.available:
            raise RuntimeError("No connections available (pool exhausted)")
        conn = self.available.pop()
        self.in_use.add(conn)
        return conn

    def release(self, conn: Connection) -> None:
        if conn in self.in_use:
            self.in_use.remove(conn)
            self.available.append(conn)

    @property
    def stats(self) -> dict:
        return {
            "available": len(self.available),
            "in_use": len(self.in_use),
            "total": len(self.available) + len(self.in_use),
        }


class DatabaseClientBuggy:
    def __init__(self, pool: ConnectionPoolBuggy):
        self.pool = pool

    def execute_query(self, query: str) -> list[dict]:
        """Execute a query and return results."""
        conn = self.pool.acquire()
        # BUG: If execute() raises, release() is never called
        result = conn.execute(query)
        self.pool.release(conn)
        return result

    def execute_batch(self, queries: list[str]) -> list[list[dict]]:
        """Execute multiple queries."""
        results = []
        for query in queries:
            conn = self.pool.acquire()
            # BUG: Same issue - exception causes leak
            result = conn.execute(query)
            results.append(result)
            self.pool.release(conn)
        return results


# ============================================================================
# FIXED CODE - expected solution
# ============================================================================

FIXED_CODE = '''
from contextlib import contextmanager

class ConnectionPool:
    # ... same as before ...

    @contextmanager
    def connection(self):
        """Context manager for safe connection handling."""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)


class DatabaseClient:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def execute_query(self, query: str) -> list[dict]:
        """Execute a query and return results."""
        with self.pool.connection() as conn:  # FIX: use context manager
            return conn.execute(query)

    def execute_batch(self, queries: list[str]) -> list[list[dict]]:
        """Execute multiple queries."""
        results = []
        for query in queries:
            with self.pool.connection() as conn:  # FIX: use context manager
                results.append(conn.execute(query))
        return results
'''


class ConnectionPoolFixed:
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.available: list[Connection] = []
        self.in_use: set[Connection] = set()

        for i in range(max_connections):
            self.available.append(Connection(id=i))

    def acquire(self) -> Connection:
        if not self.available:
            raise RuntimeError("No connections available (pool exhausted)")
        conn = self.available.pop()
        self.in_use.add(conn)
        return conn

    def release(self, conn: Connection) -> None:
        if conn in self.in_use:
            self.in_use.remove(conn)
            self.available.append(conn)

    @contextmanager
    def connection(self) -> Generator[Connection, None, None]:
        """Context manager for safe connection handling."""
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    @property
    def stats(self) -> dict:
        return {
            "available": len(self.available),
            "in_use": len(self.in_use),
            "total": len(self.available) + len(self.in_use),
        }


class DatabaseClientFixed:
    def __init__(self, pool: ConnectionPoolFixed):
        self.pool = pool

    def execute_query(self, query: str) -> list[dict]:
        """Execute a query and return results."""
        with self.pool.connection() as conn:
            return conn.execute(query)

    def execute_batch(self, queries: list[str]) -> list[list[dict]]:
        """Execute multiple queries."""
        results = []
        for query in queries:
            with self.pool.connection() as conn:
                results.append(conn.execute(query))
        return results


# ============================================================================
# TEST HARNESS
# ============================================================================


def test_resource_leak(use_fixed: bool = False) -> tuple[bool, str]:
    """Test connection pool behavior under exceptions.

    Returns (passed, message) tuple.
    """
    errors = []

    if use_fixed:
        pool = ConnectionPoolFixed(max_connections=3)
        client = DatabaseClientFixed(pool)
    else:
        pool = ConnectionPoolBuggy(max_connections=3)
        client = DatabaseClientBuggy(pool)

    initial_stats = pool.stats.copy()
    print(f"Initial pool state: {initial_stats}")

    # Test 1: Normal queries should work
    try:
        result = client.execute_query("SELECT * FROM users")
        if not result:
            errors.append("Test 1 FAIL: Normal query returned no results")
    except Exception as e:
        errors.append(f"Test 1 FAIL: Normal query raised {e}")

    # Test 2: Failed query should NOT leak connection
    try:
        client.execute_query("SELECT FAIL FROM nowhere")
    except ValueError:
        pass  # Expected
    except Exception as e:
        errors.append(f"Test 2 FAIL: Unexpected error type: {e}")

    mid_stats = pool.stats.copy()
    print(f"After failed query: {mid_stats}")

    if mid_stats["available"] != initial_stats["available"]:
        errors.append(
            f"Test 2 FAIL: Connection leaked! "
            f"Available: {mid_stats['available']} (expected {initial_stats['available']})"
        )

    # Test 3: Multiple failed queries should not exhaust pool
    pool_exhausted = False
    for i in range(5):  # More than pool size
        try:
            client.execute_query(f"FAIL query {i}")
        except ValueError:
            pass  # Expected query failure
        except RuntimeError as e:
            if "pool exhausted" in str(e).lower() or "no connections" in str(e).lower():
                pool_exhausted = True
                break
            raise

    final_stats = pool.stats.copy()
    print(f"After failed queries: {final_stats}")

    if pool_exhausted:
        errors.append("Test 3 FAIL: Pool exhausted due to connection leaks!")
    elif final_stats["available"] != initial_stats["available"]:
        errors.append(
            f"Test 3 FAIL: Multiple leaks! "
            f"Available: {final_stats['available']} (expected {initial_stats['available']})"
        )

    # Test 4: Batch with one failure should not leak
    # Skip if pool already exhausted from Test 3
    if not pool_exhausted:
        try:
            client.execute_batch(["SELECT 1", "FAIL", "SELECT 2"])
        except ValueError:
            pass
        except RuntimeError:
            errors.append("Test 4 FAIL: Pool exhausted during batch!")

        batch_stats = pool.stats.copy()
        print(f"After failed batch: {batch_stats}")

        if batch_stats["available"] != initial_stats["available"]:
            errors.append(
                f"Test 4 FAIL: Batch leaked connections! "
                f"Available: {batch_stats['available']} (expected {initial_stats['available']})"
            )
    else:
        print("Test 4 SKIPPED: Pool already exhausted")

    if errors:
        return False, "\n".join(errors)
    return True, "All tests passed! No connection leaks detected."


# ============================================================================
# SCORING HELPERS
# ============================================================================

BUG_KEYWORDS = [
    "resource leak",
    "connection leak",
    "memory leak",
    "not released",
    "never released",
    "exception",
    "try/finally",
    "try-finally",
    "context manager",
    "with statement",
    "cleanup",
    "finally block",
    "error handling",
    "exception safety",
]

FIX_KEYWORDS = [
    "contextmanager",
    "context manager",
    "with statement",
    "@contextmanager",
    "try:",
    "finally:",
    "__enter__",
    "__exit__",
    "yield conn",
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

    passed, msg = test_resource_leak(use_fixed)
    print(msg)
    print("-" * 40)

    if passed:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)
