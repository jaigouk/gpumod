from collections import deque
from typing import Optional, Tuple, List, Dict


class JobQueue:
    """A simple priority queue for jobs with FIFO ordering within the same priority."""

    def __init__(self) -> None:
        # Index 0 -> priority 0 (normal), 1 -> priority 1 (high), 2 -> priority 2 (critical)
        self._queues: List[deque] = [deque() for _ in range(3)]

    def add_job(self, job: str, job_data: Dict, priority: int = 0) -> None:
        """Add a job to the queue.

        Args:
            job: Identifier for the job.
            job_data: Data associated with the job.
            priority: Numeric priority (0 = normal, 1 = high, 2 = critical). Default is 0.
        """
        if not isinstance(priority, int) or priority < 0:
            raise ValueError("Priority must be a non‑negative integer")
        # Ensure the priority index exists
        if priority >= len(self._queues):
            # Extend the internal storage up to the required priority
            for i in range(len(self._queues), priority + 1):
                self._queues.append(deque())
        self._queues[priority].append((job, job_data))

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """Return the highest‑priority job available, or None if the queue is empty.

        The returned value is a tuple (job_id, data). Higher numeric priority
        is processed first; jobs with the same priority retain FIFO order.
        """
        for priority in range(len(self._queues) - 1, -1, -1):
            if self._queues[priority]:
                job, data = self._queues[priority].popleft()
                return job, data
        return None