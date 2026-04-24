# queue/core.py

from dataclasses import dataclass
from typing import Callable, Any, Optional

# --- Job Definition ---

@dataclass(order=True)
class Job:
    """
    Represents a single unit of work.
    The 'priority' attribute is used for ordering in the PriorityQueue.
    """
    priority: int
    job_id: str
    payload: Any
    # Custom comparison is needed for heapq when priorities are equal
    # We use job_id as a tie-breaker to ensure consistent ordering
    _tie_breaker: str = "" 

    def __post_init__(self):
        # Ensure the tie-breaker is unique for stable sorting
        if not self._tie_breaker:
            import uuid
            self._tie_breaker = str(uuid.uuid4())

    def __lt__(self, other):
        # Custom comparison logic for the heap
        return (self.priority, self._tie_breaker) < (other.priority, other._tie_breaker)

# --- Job Queue Implementation ---

class JobQueue:
    """
    Manages the collection of jobs, utilizing a PriorityQueue internally.
    """
    def __init__(self):
        # The actual queue implementation will be imported from priority.py
        from .priority import PriorityQueue
        self._queue = PriorityQueue()

    def enqueue(self, job: Job):
        """Adds a job to the queue."""
        self._queue.push(job)
        print(f"Job {job.job_id} enqueued with priority {job.priority}.")

    def dequeue(self) -> Optional[Job]:
        """Removes and returns the highest priority job."""
        return self._queue.pop()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.is_empty()