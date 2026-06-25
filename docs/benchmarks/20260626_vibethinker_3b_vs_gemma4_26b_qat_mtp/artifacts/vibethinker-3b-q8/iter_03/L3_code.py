import heapq
from typing import Dict, Tuple, Optional

class JobQueue:
    def __init__(self) -> None:
        # Min‑heap where we store (-priority, insertion_order, job_id, job_data)
        self._heap: List[Tuple[int, int, str, Dict]] = []
        self._order: int = 0

    def add_job(self, job_id: str, job_data: Dict, priority: int = 0) -> None:
        """
        Add a job to the queue.
        `priority` is optional; default is 0.
        Higher numeric priority is processed first (0 = normal, 1 = high, 2 = critical).
        """
        # Use negative priority because heapq is a min‑heap.
        heapq.heappush(self._heap, (-priority, self._order, job_id, job_data))
        self._order += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """
        Return the highest‑priority job as a (job_id, job_data) tuple,
        or None if the queue is empty.
        """
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return (job_id, job_data)