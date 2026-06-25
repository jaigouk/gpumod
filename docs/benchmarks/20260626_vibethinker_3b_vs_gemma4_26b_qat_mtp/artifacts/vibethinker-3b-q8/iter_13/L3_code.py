from collections import deque
from typing import Any, Dict, Optional, Tuple


class JobQueue:
    """A simple job queue with priority scheduling (0 = normal, 1 = high, 2 = critical)."""

    def __init__(self) -> None:
        # One deque per priority level; maintain FIFO within each level.
        self._queues: dict[int, deque[Tuple[str, Dict]]] = {
            0: deque(),
            1: deque(),
            2: deque()
        }

    def add_job(
        self,
        job: str,
        job_info: Dict[Any, Any],
        priority: int = 0,
    ) -> None:
        """Add a job to the queue.

        Args:
            job: Identifier for the job.
            job_info: Dictionary containing the job's details.
            priority: Priorities: 0 = normal, 1 = high, 2 = critical (default=0).
        """
        if not 0 <= priority <= 2:
            raise ValueError("Priority must be 0, 1, or 2.")
        self._queues[priority].append((job, job_info))

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """Return the highest‑priority job currently in the queue.

        The method respects FIFO order for jobs with the same priority.

        Returns:
            (job, job_info) for the next job, or None if the queue is empty.
        """
        # Check priorities from highest (2) down to lowest (0)
        for priority in range(2, -1, -1):
            if self._queues[priority]:
                job, job_info = self._queues[priority].popleft()
                return job, job_info
        return None