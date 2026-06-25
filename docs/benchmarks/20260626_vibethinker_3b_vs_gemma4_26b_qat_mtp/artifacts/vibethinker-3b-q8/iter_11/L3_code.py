from collections import deque
from typing import Any, Dict, Tuple, List, Optional

class JobQueue:
    def __init__(self) -> None:
        # One deque per priority level (0, 1, 2). Index corresponds to priority.
        self._queues: List[deque] = [deque() for _ in range(3)]

    def add_job(self, job_name: str, job_data: Dict[Any, Any], priority: int = 0) -> None:
        """Add a job to the queue. Jobs with the same priority retain FIFO order."""
        if priority < 0:
            raise ValueError("Priority must be a non‑negative integer")
        # Ensure we have a deque for the specified priority.
        if priority >= len(self._queues):
            # Extend the list with empty deques if a higher priority is used.
            for _ in range(len(self._queues), priority + 1):
                self._queues.append(deque())
        self._queues[priority].append((job_name, job_data))

    def get_next_job(self) -> Tuple[str, Dict[str, Any]] | None:
        """Return the highest‑priority job (largest numeric priority) or None if empty."""
        for p in range(len(self._queues) - 1, -1, -1):
            if self._queues[p]:
                job_name, job_data = self._queues[p].popleft()
                return job_name, job_data
        return None