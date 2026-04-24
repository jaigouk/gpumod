# queue/core.py
from dataclasses import dataclass, field
from typing import Any, Callable, List
import heapq
from queue.priority import PriorityQueue # Import the specialized queue

@dataclass(order=True)
class Job:
    """Represents a single unit of work."""
    # Priority is the first element for heapq comparison
    priority: int = field(compare=True) 
    # Unique identifier for the job
    job_id: str = field(compare=False)
    # The function to execute
    func: Callable[..., Any] = field(compare=False)
    # Arguments for the function
    args: tuple = field(compare=False)
    # Metadata for retries
    attempts: int = field(default=0, compare=False)
    max_attempts: int = field(default=3, compare=False)


class JobQueue:
    """
    Manages the collection of jobs, utilizing a PriorityQueue 
    to ensure high-priority tasks are processed first.
    """
    def __init__(self):
        # Use the PriorityQueue implementation
        self._queue = PriorityQueue()

    def add_job(self, job: Job):
        """Adds a job to the queue."""
        self._queue.put(job)
        print(f"[QUEUE] Job {job.job_id} added with priority {job.priority}.")

    def get_next_job(self) -> Job | None:
        """Retrieves and removes the highest priority job."""
        try:
            return self._queue.get()
        except IndexError:
            return None

    def is_empty(self) -> bool:
        """Checks if the queue contains any jobs."""
        return self._queue.empty()

    def size(self) -> int:
        """Returns the number of jobs currently in the queue."""
        return self._queue.qsize()