from collections import deque
from typing import Tuple, Optional

class JobQueue:
    def __init__(self) -> None:
        # One deque per priority level: 0 (normal), 1 (high), 2 (critical)
        self._queues: List[deque[Tuple[str, dict]]] = [deque() for _ in range(3)]

    def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
        # Clamp priority to the valid range [0, 2] for safety
        priority = max(0, min(priority, 2))
        self._queues[priority].append((job_name, job_data))

    def get_next_job(self) -> Tuple[str, dict] | None:
        # Search from highest priority (2) down to 0
        for p in range(2, -1, -1):
            if self._queues[p]:
                job_name, job_data = self._queues[p].popleft()
                return (job_name, job_data)
        return None