import heapq
from typing import Dict, Tuple, List, Optional

class JobQueue:
    def __init__(self) -> None:
        # Min‑heap storing: (-priority, insertion_order, job_id, job_data)
        self._heap: List[Tuple[int, int, str, Dict]] = []
        self._counter: int = 0

    def add_job(self, job_id: str, job_data: Dict, priority: int = 0) -> None:
        """Add a job to the queue. Higher numeric priority is processed first."""
        # Use a monotonic counter to enforce FIFO ordering for equal priorities.
        heapq.heappush(self._heap, (-priority, self._counter, job_id, job_data))
        self._counter += 1

    def get_next_job(self) -> Optional[Tuple[str, Dict]]:
        """Return the highest‑priority job available, or None if the queue is empty."""
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return job_id, job_data