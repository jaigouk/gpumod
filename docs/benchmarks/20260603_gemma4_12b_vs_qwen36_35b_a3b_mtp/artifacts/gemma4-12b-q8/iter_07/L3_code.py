import heapq
from typing import Tuple, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = []
        self._counter = 0

    def add_job(self, job_id: str, job_data: Dict, priority: int = 0) -> None:
        # heapq is a min-heap. To treat higher priority numbers as "smaller" 
        # (higher priority), we negate the priority value.
        # self._counter ensures FIFO order for jobs with the same priority.
        heapq.heappush(self._queue, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        if not self._queue:
            return None

        _, _, job_id, job_data = heapq.heappop(self._queue)
        return (job_id, job_data)