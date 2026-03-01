import heapq
from __future__ import annotations

class JobQueue:
    def __init__(self):
        self._heap = []
        self._sequence = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        heapq.heappush(self._heap, (-priority, self._sequence, job_id, data))
        self._sequence += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._heap:
            return None
        _, _, job_id, data = heapq.heappop(self._heap)
        return (job_id, data)