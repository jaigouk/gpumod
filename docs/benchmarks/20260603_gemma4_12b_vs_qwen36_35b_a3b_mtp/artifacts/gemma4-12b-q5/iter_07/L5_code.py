from dataclasses import dataclass, field
    from typing import Callable, Any, List
    import uuid

    @dataclass
    class Job:
        task: Callable
        args: tuple = field(default_factory=tuple)
        kwargs: dict = field(default_factory=dict)
        id: str = field(default_factory=lambda: str(uuid.uuid4()))
        priority: int = 0

    class JobQueue:
        def __init__(self):
            self._jobs: List[Job] = []
        def push(self, job: Job): self._jobs.append(job)
        def pop(self) -> Job: return self._jobs.pop(0)
        def __len__(self): return len(self._jobs)