# queue/core.py
from dataclasses import dataclass, field
from typing import Callable, Any
from .priority import PriorityQueue
import time

@dataclass(order=True)
class Job:
    """
    Represents a single unit of work in the queue.
    The 'priority' field is used for sorting.
    """
    priority: int = field(compare=True)
    id: str = field(compare=False)
    payload: Any = field(compare=False)
    max_retries: int = 3
    attempt_count: int = 0
    
    def __str__(self):
        return f"Job(id={self.id}, priority={self.priority}, attempts={self.attempt_count})"

class JobQueue:
    """
    Manages the collection of jobs using a PriorityQueue.
    This is the primary interface for adding and retrieving jobs.
    """
    def __init__(self):
        # Use the imported PriorityQueue implementation
        self._queue = PriorityQueue()

    def enqueue(self, job: Job):
        """Adds a job to the queue."""
        if not isinstance(job, Job):
            raise TypeError("Input must be a Job object.")
        self._queue.put(job)
        print(f"[{time.strftime('%H:%M:%S')}] Enqueued job: {job.id} (P={job.priority})")

    def dequeue(self) -> Job | None:
        """Removes and returns the highest priority job."""
        return self._queue.get()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()

    def size(self) -> int:
        """Returns the number of jobs currently in the queue."""
        return self._queue.size()