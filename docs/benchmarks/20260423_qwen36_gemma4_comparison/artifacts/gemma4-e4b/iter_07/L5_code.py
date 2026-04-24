from dataclasses import dataclass, field
from typing import Callable, Any, List
from queue.priority import PriorityQueue

@dataclass(order=True)
class Job:
    """
    Represents a unit of work in the queue.
    The 'priority' field is used for sorting in the PriorityQueue.
    """
    priority: int = field(compare=True)
    job_id: str = field(compare=False)
    func: Callable[..., Any] = field(compare=False)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: dict = field(compare=False, default_factory=dict)
    retries: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)

    def __str__(self):
        return f"Job(ID={self.job_id}, Priority={self.priority}, Retries={self.retries})"


class JobQueue:
    """
    The primary interface for managing and retrieving jobs.
    Uses PriorityQueue internally.
    """
    def __init__(self):
        # The PriorityQueue manages the actual heap structure
        self._queue = PriorityQueue()

    def enqueue(self, job: Job):
        """Adds a job to the queue."""
        self._queue.put(job)
        print(f"[Queue] Enqueued job: {job.job_id} (P={job.priority})")

    def dequeue(self) -> Job | None:
        """Removes and returns the highest priority job."""
        return self._queue.get()

    def empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()

    def size(self) -> int:
        """Returns the number of jobs currently in the queue."""
        return self._queue.qsize()