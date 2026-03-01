from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

@dataclass
class Job:
    """Represents a single unit of work."""
    id: str
    payload: Any
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    
    def __lt__(self, other):
        # Helper for comparison if needed in core logic
        return self.priority > other.priority

class JobQueue:
    """A standard FIFO (First-In-First-Out) Queue."""
    
    def __init__(self):
        self._queue: List[Job] = []
    
    def enqueue(self, job: Job) -> None:
        """Add a job to the end of the queue."""
        self._queue.append(job)
        job.status = "pending"
    
    def dequeue(self) -> Optional[Job]:
        """Remove and return the first job from the queue."""
        if self.is_empty():
            return None
        job = self._queue.pop(0)
        job.status = "processing"
        return job
    
    def peek(self) -> Optional[Job]:
        """Return the first job without removing it."""
        if self.is_empty():
            return None
        return self._queue[0]
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0
    
    def size(self) -> int:
        return len(self._queue)
    
    def clear(self) -> None:
        self._queue.clear()