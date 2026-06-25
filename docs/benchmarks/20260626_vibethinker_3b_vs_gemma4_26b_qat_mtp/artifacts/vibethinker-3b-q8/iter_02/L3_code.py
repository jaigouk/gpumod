from collections import deque
from typing import Dict, Tuple, Optional


class JobQueue:
    def __init__(self) -> None:
        # Index 0: normal, 1: high, 2: critical
        self._queues: List[deque[Tuple[str, Dict]]] = [deque() for _ in range(3)]

    def add_job(self, job: str, job_info: Dict, priority: int = 0) -> None:
        """Add a job to the queue.

        Args:
            job: The identifier or name of the job.
            job_info: Additional information about the job.
            priority: Prioritization level (0=normal, 1=high, 2=critical). Default is 0.
        """
        if not 0 <= priority <= 2:
            raise ValueError("Priority must be 0, 1, or 2")
        self._queues[priority].append((job, job_info))

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """Return the highest‑priority job currently in the queue.

        Returns:
            A tuple (job, info) of the next job, or None if the queue is empty.
        """
        for p in (2, 1, 0):
            if self._queues[p]:
                job, info = self._queues[p].popleft()
                return job, info
        return None