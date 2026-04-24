from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
from collections import deque
from datetime import datetime

@dataclass
class Job:
    """Represents a unit of work to be processed."""
    id: str = field(default_factory=lambda: str(uuid4()))
    payload: Any = None
    priority: int = 0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0

class JobQueue:
    """Basic FIFO job queue implementation."""
    def __init__(self) -> None:
        self._queue: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        job.status = "queued"
        self._queue.append(job)

    def dequeue(self) -> Optional[Job]:
        if not self._queue:
            return None
        job = self._queue.popleft()
        job.status = "processing"
        return job

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def size(self) -> int:
        return len(self._queue)