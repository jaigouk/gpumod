from collections import deque
from typing import Dict, Tuple, Optional

class JobQueue:
    def __init__(self) -> None:
        # priorities: 0 = normal, 1 = high, 2 = critical
        self._queues: list[deque[Tuple[str, Dict]]] = [deque() for _ in range(3)]

    def add_job(self, job_id: str, job_data: Dict, priority: int = 0) -> None:
        if not 0 <= priority <= 2:
            raise ValueError("Priority must be 0, 1, or 2")
        self._queues[priority].append((job_id, job_data))

    def get_next_job(self) -> Tuple[str, Dict] | None:
        for p in range(2, -1, -1):  # check critical (2) down to normal (0)
            if self._queues[p]:
                job_id, job_data = self._queues[p].popleft()
                return (job_id, job_data)
        return None