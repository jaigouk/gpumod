from dataclasses import dataclass, field
        from typing import Any

        @dataclass(order=True)
        class Job:
            priority: int
            data: Any = field(compare=False)

        class JobQueue:
            def __init__(self):
                self._jobs = []
            def push(self, job: Job): self._jobs.append(job)
            def pop(self) -> Job: return self._jobs.pop(0)
            def __len__(self): return len(self._jobs)