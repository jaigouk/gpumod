import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque

@dataclass
class Job:
    """Represents a unit of work to be processed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    priority: int = 0
    status: str = "pending"
    retries: int = 0
    max_retries: int = 3

class JobQueue:
    """Basic FIFO job queue implementation."""
    def __init__(self) -> None:
        self._queue: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        self._queue.append(job)

    def dequeue(self) -> Optional[Job]:
        if self._queue:
            return self._queue.popleft()
        return None

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0