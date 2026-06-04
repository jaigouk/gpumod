from dataclasses import dataclass, field
        from abc import ABC, abstractmethod
        from typing import Any

        @dataclass(order=True)
        class Job:
            priority: int
            job_id: str = field(compare=False)
            payload: Any = field(compare=False)

        class JobQueue(ABC):
            @abstractmethod
            def push(self, job: Job): pass
            @abstractmethod
            def pop(self) -> Job: pass
            @abstractmethod
            def is_empty(self) -> bool: pass