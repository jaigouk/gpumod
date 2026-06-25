from collections import deque
from typing import Dict, Optional, Tuple


class JobQueue:
    """Priority queue for scheduling jobs with FIFO ordering within the same priority.

    Priorities:
        0 – normal (default)
        1 – high
        2 – critical
    """

    def __init__(self) -> None:
        # One deque per priority level (0, 1, 2)
        self._queues: list[deque[Tuple[str, Dict]]] = [deque() for _ in range(3)]

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        """Add a job to the queue.

        Args:
            job_id: Identifier for the job.
            job_data: Dictionary containing the job's data.
            priority: Prioritization level (0 = normal, 1 = high, 2 = critical).
        """
        if not (0 <= priority <= 2):
            raise ValueError("Priority must be 0, 1, or 2")
        self._queues[priority].append((job_id, job_data))

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        """Return the highest‑priority job available.

        Returns:
            A tuple (job_id, job_data) of the next job to execute, or None if the queue is empty.
        """
        for priority in range(2, -1, -1):
            while self._queues[priority]:
                job_id, job_data = self._queues[priority].popleft()
                return (job_id, job_data)
        return None