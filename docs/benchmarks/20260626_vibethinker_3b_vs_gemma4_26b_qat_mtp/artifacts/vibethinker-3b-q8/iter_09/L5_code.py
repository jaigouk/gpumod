from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional, List
import heapq

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> Tuple[bool, int]:
        attempt = 0
        while attempt < self.max_attempts:
            try:
                fn(data)
                return (True, attempt + 1)
            except Exception:
                attempt += 1
                continue
        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self._heap: List[Tuple[int, int, Any]] = []
        self._index: int = 0
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        heapq.heappush(self._heap, (-priority, self._index, job))
        self._index += 1

    def process_job(self, job_id: str, processor: Callable[[Any], Any]) -> bool:
        if not self._heap:
            return False
        _, _, job = heapq.heappop(self._heap)
        success, _ = self._retry_policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._heap:
            return None
        _, _, job = self._heap[0]
        return (job.id, job.data)