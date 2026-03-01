from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Optional
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    """Represents a unit of work to be processed."""
    id: str
    payload: Any
    priority: int = 0  # Higher number = higher priority
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3

    def __repr__(self):
        return f"Job(id={self.id}, status={self.status.value}, priority={self.priority})"

class JobQueue:
    """A basic FIFO queue for Jobs."""
    
    def __init__(self):
        self._queue: Queue[Job] = Queue()

    def enqueue(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.PENDING
        self._queue.put(job)

    def dequeue(self) -> Optional[Job]:
        """Remove and return the next job from the queue."""
        try:
            return self._queue.get_nowait()
        except:
            return None

    def peek(self) -> Optional[Job]:
        """Return the next job without removing it."""
        if self._queue.empty():
            return None
        return self._queue.queue[0]

    def size(self) -> int:
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        return self._queue.empty()