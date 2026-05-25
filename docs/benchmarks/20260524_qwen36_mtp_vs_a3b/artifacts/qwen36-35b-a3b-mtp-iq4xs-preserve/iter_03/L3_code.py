import heapq
from typing import Optional, Tuple, Dict

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, name: str, job_data: dict, priority: int = 0) -> None:
        # Store as (-priority, counter, name, job_data)
        # Negate priority because heapq is a min-heap
        heapq.heappush(self._queue, (-priority, self._counter, name, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._queue:
            return None
        _, _, name, job_data = heapq.heappop(self._queue)
        return (name, job_data)