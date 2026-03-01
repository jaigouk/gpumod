# queue/core.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List
from collections import deque
import threading

@dataclass
class Job:
    """Represents a unit of work to be processed."""
    id: str
    payload: Any
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    priority: int = 0  # 0 is lowest priority by default

    def __repr__(self) -> str:
        return f"Job(id={self.id}, status={self.status})"

class JobQueue:
    """A standard FIFO queue for Job objects."""
    
    def __init__(self):
        self._queue: deque = deque()
        self._lock = threading.Lock()

    def enqueue(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        with self._lock:
            job.status = "pending"
            self._queue.append(job)

    def dequeue(self) -> Optional[Job]:
        """Remove and return the first job in the queue."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def peek(self) -> Optional[Job]:
        """Return the first job without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def size(self) -> int:
        """Return the number of jobs in the queue."""
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.size() == 0