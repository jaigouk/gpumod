import heapq
from typing import Optional, Tuple

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        heapq.heappush(self._heap, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return (job_id, job_data)