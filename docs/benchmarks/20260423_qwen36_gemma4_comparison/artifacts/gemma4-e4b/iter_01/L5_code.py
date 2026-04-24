# queue/core.py

import dataclasses
from typing import Any, Optional

@dataclasses.dataclass(order=True)
class Job:
    """
    Represents a single unit of work.
    The 'priority' field is used for sorting in the PriorityQueue.
    """
    priority: int = 0  # Lower number means higher priority
    job_id: str = dataclasses.field(compare=False)
    task: Any = dataclasses.field(compare=False)
    retries_attempted: int = 0

    def __post_init__(self):
        # Ensure job_id is set
        if not self.job_id:
            raise ValueError("Job must have a unique job_id.")

class JobQueue:
    """
    The main interface for managing and processing jobs.
    It relies on an underlying priority queue implementation.
    """
    def __init__(self, priority_queue):
        self._queue = priority_queue

    def enqueue(self, job: Job):
        """Adds a job to the queue."""
        print(f"Queue: Enqueuing job {job.job_id} (Priority: {job.priority})")
        self._queue.put(job)

    def dequeue(self) -> Optional[Job]:
        """Retrieves the highest priority job from the queue."""
        try:
            job = self._queue.get()
            print(f"Queue: Dequeued job {job.job_id}")
            return job
        except IndexError:
            # Handle empty queue case
            return None

    def is_empty(self) -> bool:
        """Checks if the queue contains any jobs."""
        return self._queue.empty()

# Note: The actual PriorityQueue implementation is imported from queue.priority
# and instantiated in queue/__init__.py