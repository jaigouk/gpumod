import heapq
from typing import Any, Dict, Tuple, Optional, List

class JobQueue:
    def __init__(self) -> None:
        # Each entry: (-priority, timestamp, job_id, job_data)
        self._heap: List[Tuple[int, int, str, Dict[str, Any]]] = []
        self._time: int = 0

    def add_job(self, job_id: str, job_data: Dict[str, Any], priority: int = 0) -> None:
        """Add a job to the queue with an optional priority."""
        self._time += 1
        # Negative priority makes heapq treat higher priority as smaller key
        heapq.heappush(self._heap, (-priority, self._time, job_id, job_data))

    def get_next_job(self) -> Tuple[str, Dict[str, Any]] | None:
        """Return the highest‑priority job currently in the queue."""
        if not self._heap:
            return None
        _, _, job_id, job_data = heapq.heappop(self._heap)
        return job_id, job_data