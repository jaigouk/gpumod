from dataclasses import dataclass
        from typing import Any, List
        from abc import ABC, abstractmethod

        @dataclass
        class Job:
            id: str
            task: str
            data: Any
            priority: int = 0
            retries: int = 0

        class JobQueue(ABC):
            def __init__(self):
                self._queue = []
            @abstractmethod
            def push(self, job: Job): ...
            @abstractmethod
            def pop(self) -> Job: ...