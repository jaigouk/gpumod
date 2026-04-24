# queue/core.py

import dataclasses
from typing import Callable, Any, Optional
from .priority import PriorityQueue

@dataclasses.dataclass(order=True)
class Job:
    """Represents a unit of work to be executed."""
    # Priority is used by the PriorityQueue (lower number = higher priority)
    priority: int = 0
    # The job ID/unique identifier
    job_id: str = dataclasses.field(default_factory=lambda: str(hash(object())))
    # The function to execute
    func: Callable[..., Any]
    # Arguments for the function
    args: tuple = dataclasses.field(default_factory=tuple)
    # Metadata for tracking retries
    attempt_count: int = 0

    def execute(self):
        """Executes the job's function."""
        print(f"Executing Job {self.job_id} (Priority: {self.priority}, Attempt: {self.attempt_count})")
        return self.func(*self.args)

class JobQueue:
    """
    Manages the collection and retrieval of jobs using a PriorityQueue.
    """
    def __init__(self):
        # Use the dedicated PriorityQueue implementation
        self._queue = PriorityQueue()
        print("JobQueue initialized.")

    def add_job(self, job: Job):
        """Adds a job to the queue."""
        self._queue.put(job)
        print(f"Job {job.job_id} added to queue.")

    def get_next_job(self) -> Optional[Job]:
        """Retrieves and removes the highest priority job."""
        return self._queue.get()

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self._queue.empty()

    def size(self) -> int:
        """Returns the current number of jobs in the queue."""
        return self._queue.size()