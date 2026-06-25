from collections import deque
from typing import Tuple, Optional, List, Dict

class JobQueue:
    def __init__(self) -> None:
        # One deque per priority (0, 1, 2). Index corresponds to priority.
        self._queues: List[deque[Tuple[str, Dict]]] = [deque() for _ in range(3)]

    def add_job(
        self,
        job_name: str,
        job_data: Dict,
        priority: int = 0,
    ) -> None:
        """Add a job to the queue.

        Args:
            job_name: Name of the job (str).
            job_data: Data attached to the job (dict).
            priority: Prioritization value (0 = normal, 1 = high, 2 = critical).
        """
        if not isinstance(job_name, str) or not isinstance(job_data, dict):
            raise TypeError("job_name must be a string and job_data must be a dict")
        if not 0 <= priority < 3:
            raise ValueError("priority must be 0, 1, or 2")
        self._queues[priority].append((job_name, job_data))

    def get_next_job(
        self,
        *,
        return_value: Optional[Tuple[str, Dict]] = None,
    ) -> Optional[Tuple[str, Dict]]:
        """Return the highest‑priority job without removing it, or remove and return it.

        The specification asks for `get_next_job()` that returns the job, so we
        pop the job and return it. If the queue is empty, returns None.

        Returns:
            A tuple `(job_name, job_data)` for the next job, or None if the queue is empty.
        """
        # Highest numeric priority first (2 > 1 > 0)
        for p in range(2, -1, -1):
            if self._queues[p]:
                job_name, job_data = self._queues[p].popleft()
                return job_name, job_data
        return None