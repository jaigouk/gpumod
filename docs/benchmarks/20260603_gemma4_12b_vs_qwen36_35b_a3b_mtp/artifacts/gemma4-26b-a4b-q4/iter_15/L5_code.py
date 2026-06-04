from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=False) # We will handle comparison manually for priority
        class Job:
            id: str
            payload: Any
            priority: int = 0

            def __lt__(self, other):
                # For PriorityQueue (heapq is a min-heap, so higher priority = lower number)
                return self.priority < other.priority

        class JobQueue:
            def __init__(self):
                self._jobs: List[Job] = []

            def enqueue(self, job: Job):
                self._jobs.append(job)

            def dequeue(self) -> Job:
                return self._jobs.pop(0) if self._jobs else None

            def __len__(self):
                return len(self._jobs)