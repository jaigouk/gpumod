# queue/core.py

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from queue.priority import PriorityQueue

@dataclass(order=True)
class Job:
    """
    Represents a unit of work to be executed.
    The 'priority' field is used by the PriorityQueue.
    """
    priority: int = field(compare=True)
    job_id: str = field(compare=False)
    task: Callable[..., Any] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    max_retries: int = field(default=3, compare=False)
    current_attempts: int = field(default=0, compare=False)

    def __post_init__(self):
        # Ensure priority is an integer for comparison
        if not isinstance(self.priority, int):
            raise TypeError("Job priority must be an integer.")


class JobQueue:
    """
    The main queue manager that holds and processes jobs.
    It relies on PriorityQueue for ordering.
    """
    def __init__(self):
        # The PriorityQueue handles the heap structure
        self._queue = PriorityQueue()
        print("JobQueue initialized.")

    def add_job(self, job: Job):
        """Adds a job to the queue."""
        self._queue.put(job)
        print(f"Job '{job.job_id}' added with priority {job.priority}.")

    def get_next_job(self) -> Optional[Job]:
        """Retrieves and removes the highest priority job."""
        return self._queue.get()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()