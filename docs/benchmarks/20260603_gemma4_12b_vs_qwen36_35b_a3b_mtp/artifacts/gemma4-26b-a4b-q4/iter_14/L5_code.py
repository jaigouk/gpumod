from dataclasses import dataclass, field
        from typing import Any, Deque
        from collections import deque

        @dataclass(order=False) # We handle order in PriorityQueue
        class Job:
            id: str
            priority: int
            data: Any

        class JobQueue:
            def __init__(self):
                self._queue: Deque[Job] = deque()
            def enqueue(self, job: Job): self._queue.append(job)
            def dequeue(self) -> Job: return self._queue.popleft()
            def is_empty(self) -> bool: return len(self._queue) == 0
            def size(self) -> int: return len(self._queue)