from dataclasses import dataclass, field
        from typing import Any, Optional
        from abc import ABC, abstractmethod

        @dataclass
        class Job:
            id: str
            payload: Any
            priority: int = 0

        class JobQueue(ABC):
            def __init__(self):
                self._queue = []
            def push(self, job: Job): ...
            def pop(self) -> Optional[Job]: ...