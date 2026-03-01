import heapq
from typing import Dict, Optional, Tuple

class JobQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_job(self, job_id: str, data: Dict, priority: int = 0) -> None:
        # Use negative priority for max-heap behavior (higher priority number = processed first)
        # Use counter to maintain FIFO order for same priority
        heapq.heappush(self._heap, (-priority, self._counter, job_id, data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        if not self._heap:
            return None
        _, _, job_id, data = heapq.heappop(self._heap)
        return (job_id, data)