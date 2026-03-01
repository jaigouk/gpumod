import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from collections import deque


@dataclass
class Job:
    """Represents a unit of work to be processed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    priority: int = 0  # Higher number = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"

    def __post_init__(self):
        if self.status not in ["pending", "completed", "failed"]:
            raise ValueError("Status must be pending, completed, or failed")


class JobQueue:
    """A standard FIFO queue for jobs."""

    def __init__(self):
        self._queue: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        self._queue.append(job)

    def dequeue(self) -> Optional[Job]:
        """Remove and return the oldest job."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def peek(self) -> Optional[Job]:
        """Return the oldest job without removing it."""
        if not self._queue:
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)