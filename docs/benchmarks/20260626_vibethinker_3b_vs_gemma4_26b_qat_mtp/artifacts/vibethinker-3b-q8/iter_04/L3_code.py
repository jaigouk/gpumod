from collections import deque
from typing import Dict, Tuple, Optional

class JobQueue:
    """Priority queue for jobs with FIFO order within the same priority."""
    # Defined priority levels: 0 = normal, 1 = high, 2 = critical
    _MAX_PRIORITY = 2

    def __init__(self) -> None:
        # One deque per priority level (0 .. _MAX_PRIORITY)
        self._queues: list[deque[Tuple[str, Dict]]] = [
            deque() for _ in range(self._MAX priority + 1)
        ]

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """Add a job.

        Args:
            job_name (str): Identifier for the job.
            job_data (dict): Data associated with the job.
            priority (int, optional): Prioritization level (0 = normal,
                                       1 = high, 2 = critical). Default is 0.
        """
        if not 0 <= priority <= self._MAX_priority:
            raise ValueError(
                f"Priority must be between 0 and {self._MAX_priority}, got {priority}"
            )
        self._queues[priority].append((job_name, job_data))

    def get_next_job(self) -> Tuple[str, Dict] | None:
        """Return the highest‑priority job.

        Returns:
            (job_name, job_data) tuple if a job is available, or None if the queue is empty.
        """
        for p in range(self._MAX_priority, -1, -1):
            if self._queues[p]:
                job_name, job_data = self._queues[p].popleft()
                return job_name, job_data
        return None