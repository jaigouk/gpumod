from dataclasses import dataclass, field
        from typing import Any, List

        @dataclass(order=True)
        class Job:
            priority: int
            job_id: str = field(compare=False)
            payload: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self._queue = []
            def push(self, job: Job):
                self._queue.append(job)
            def pop(self) -> Job:
                return self._queue.pop(0)
            def is_empty(self) -> bool:
                return len(self._queue) == 0