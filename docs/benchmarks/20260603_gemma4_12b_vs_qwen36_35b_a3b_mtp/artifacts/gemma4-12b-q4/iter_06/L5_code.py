from dataclasses import dataclass, field
        from typing import Any
        from collections import deque

        @dataclass(order=False) # Order handled by PriorityQueue or manual impl
        class Job:
            id: str
            data: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3
            # Need __lt__ for PriorityQueue if using heapq
            def __lt__(self, other):
                return self.priority < other.priority

        class JobQueue:
            def __init__(self):
                self._queue = deque()
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.popleft()
            def __len__(self):
                return len(self._queue)