from __future__ import annotations
import heapq
from typing import Tuple, Optional

class JobQueue:
    def __init__(self):
        self._heap: list = []
        self._counter: int = 0

    def add_job(self, job_name: str, job_data: dict, priority: int = 0) -> None:
        heapq.heappush(self._heap, (-priority, self._counter, job_name, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_name, job_data = heapq.heappop(self._heap)
        return (job_name, job_data)