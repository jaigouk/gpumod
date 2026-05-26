from dataclasses import dataclass, field
        from typing import Optional, Callable, Any
        from .priority import PriorityQueue # Need relative import

        @dataclass
        class Job:
            id: str
            func: Callable
            payload: Any = None
            priority: int = 0
            status: str = "pending"
            retry_count: int = 0

        class JobQueue:
            def __init__(self):
                self._queue = PriorityQueue()
            # methods: add, pop, etc.