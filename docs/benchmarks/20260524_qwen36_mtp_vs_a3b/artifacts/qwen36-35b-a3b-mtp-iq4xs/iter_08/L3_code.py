import heapq
from typing import Tuple, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: dict, priority: int = 0) -> None:
        # Negate priority because heapq is a min-heap
        heapq.heappush(self._queue, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._queue:
            return None
        _, _, job_id, job_data = heapq.heappop(self._queue)
        return (job_id, job_data)