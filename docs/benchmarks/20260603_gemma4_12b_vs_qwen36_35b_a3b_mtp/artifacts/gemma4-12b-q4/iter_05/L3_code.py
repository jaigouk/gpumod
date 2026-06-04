import heapq
from typing import Dict, Tuple, Optional

class JobQueue:
    def __init__(self):
        self._queue: list[Tuple[int, int, str, dict]]] = []
        self._counter = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0):
        # Using negative priority because heapq is a min-heap
        # The counter ensures FIFO order for identical priorities
        heapq.heappush(self._queue, (-priority, self._counter, job_name, job_data))
        self._counter += 1

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._queue:
            return None

        _, _, job_name, job_data = heapq.heappop(self._queue)
        return (job_name, job_data)