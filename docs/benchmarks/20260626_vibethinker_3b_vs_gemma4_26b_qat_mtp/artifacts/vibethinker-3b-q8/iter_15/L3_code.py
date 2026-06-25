import heapq
from typing import List, Tuple, Dict

class JobQueue:
    def __init__(self) -> None:
        self._heap: List[Tuple[int, int, str, Dict]] = []
        self._counter: int = 0

    def add_job(self, job_name: str, job_data: Dict, priority: int = 0) -> None:
        """Add a job with optional priority."""
        self._counter += 1  # Increment for FIFO ordering within the same priority
        # Use negative priority so that a higher numeric priority is extracted first
        heapq.heappush(self._heap, (-priority, self._counter, job_name, job_data))

    def get_next_job(self) -> Tuple[str, Dict] | None:
        """Return the highest‑priority job currently in the queue."""
        if not self._heap:
            return None
        _, _, job_name, job_data = heapq.heappop(self._heap)
        return job_name, job_data