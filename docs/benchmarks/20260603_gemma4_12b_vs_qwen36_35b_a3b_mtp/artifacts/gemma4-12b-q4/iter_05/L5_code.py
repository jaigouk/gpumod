from dataclasses import dataclass
        from typing import Any, List

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0
            retries: int = 0
            max_retries: int = 3

        class JobQueue:
            def __init__(self):
                self._jobs: List[Job] = []
            def push(self, job: Job):
                self._jobs.append(job)
            def pop(self) -> Job:
                return self._jobs.pop(0) if self._jobs else None
            def __len__(self):
                return len(self._jobs)