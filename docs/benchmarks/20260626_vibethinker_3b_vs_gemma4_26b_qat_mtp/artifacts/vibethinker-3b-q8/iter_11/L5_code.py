from typing import Callable, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
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

    def run(self, fn: Callable[[], Any], data: Dict[str, Any]) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except:
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._heap: List[Tuple[int, int, Job]] = []
        self._counter: int = 0

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        heapq.heappush(self._heap, (-priority, self._counter, job))
        self._counter += 1

    def process_job(self, job_id: str, processor: Callable[[Dict[str, Any]], Any]) -> bool:
        if not self._heap:
            return False
        _, _, job = heapq.heappop(self._heap)
        max_attempts = job.retries + 1
        policy = RetryPolicy(max_attempts)
        success, attempts = policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._heap:
            return None
        _, _, job = self._heap[0]
        return (job.id, job.data)