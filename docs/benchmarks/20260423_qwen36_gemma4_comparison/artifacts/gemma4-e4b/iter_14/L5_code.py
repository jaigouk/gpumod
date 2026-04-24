# queue/core.py

from dataclasses import dataclass
from typing import Any, Optional
import time

@dataclass(order=True)
class Job:
    """
    Represents a single task to be executed.
    The 'priority' field is used for ordering in the PriorityQueue.
    """
    priority: int = 0  # Lower number means higher priority
    job_id: str = None
    payload: Any = None
    retries: int = 0

    def __post_init__(self):
        if self.job_id is None:
            self.job_id = f"job-{hash(self)}"

    def __str__(self):
        return f"Job(ID={self.job_id}, Priority={self.priority}, Retries={self.retries})"

class JobQueue:
    """
    The main interface for the job queue.
    It relies on an internal PriorityQueue implementation.
    """
    def __init__(self, priority_queue):
        # Dependency Injection: The queue uses a specific priority queue implementation
        self._queue = priority_queue

    def enqueue(self, job: Job):
        """Adds a job to the queue."""
        self._queue.put(job)
        print(f"[CORE] Enqueued job: {job.job_id} (P={job.priority})")

    def dequeue(self) -> Optional[Job]:
        """Removes and returns the highest priority job."""
        return self._queue.get()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()

    def size(self) -> int:
        """Returns the number of items in the queue."""
        return self._queue.qsize()