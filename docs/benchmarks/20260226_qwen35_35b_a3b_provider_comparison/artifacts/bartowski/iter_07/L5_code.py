from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional

@dataclass
class Job:
    """Represents a single unit of work."""
    id: str
    payload: Any
    priority: int = 0
    status: str = "pending"  # pending, processing, completed, failed
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    task: Optional[Callable] = None  # The actual function to execute

    def mark_processing(self):
        self.status = "processing"
        self.updated_at = datetime.now()

    def mark_completed(self):
        self.status = "completed"
        self.updated_at = datetime.now()

    def mark_failed(self):
        self.status = "failed"
        self.updated_at = datetime.now()

    def increment_retry(self):
        self.retry_count += 1
        self.updated_at = datetime.now()

class JobQueue:
    """A basic FIFO queue for managing jobs."""
    
    def __init__(self):
        self._jobs: List[Job] = []

    def enqueue(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        self._jobs.append(job)

    def dequeue(self) -> Optional[Job]:
        """Remove and return the job at the front of the queue."""
        if self._jobs:
            return self._jobs.pop(0)
        return None

    def peek(self) -> Optional[Job]:
        """Return the job at the front of the queue without removing it."""
        if self._jobs:
            return self._jobs[0]
        return None

    def size(self) -> int:
        """Return the number of jobs in the queue."""
        return len(self._jobs)

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._jobs) == 0

    def get_all(self) -> List[Job]:
        """Return a copy of all jobs in the queue."""
        return self._jobs.copy()