from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Deque
from collections import deque
import uuid

@dataclass
class Job:
    """Represents a single job in the queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    status: str = "pending"
    retries: int = 0
    created_at: float = field(default_factory=lambda: __import__('time').time())

    def mark_failed(self):
        """Increment retry count and update status."""
        self.retries += 1
        self.status = "failed"

    def mark_processed(self):
        """Update status to processed."""
        self.status = "processed"

class JobQueue:
    """A basic FIFO (First-In-First-Out) Job Queue."""

    def __init__(self):
        self._queue: Deque[Job] = deque()

    def add(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        self._queue.append(job)

    def get_next(self) -> Optional[Job]:
        """Retrieve and remove the next job from the front of the queue."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def size(self) -> int:
        """Return the number of jobs in the queue."""
        return len(self._queue)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0