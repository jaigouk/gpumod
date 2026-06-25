from dataclasses import dataclass
from typing import Callable, Dict, Any
import heapq
from collections import deque

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        attempts_made = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)

class JobQueue:
    def __init__(self):
        self.jobs_by_priority: Dict[int, deque] = {}
        self.priorities_heap: list = []
        self._job_data: Dict[str, dict] = {}
        self.retry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._job_data[job_id] = data
        if priority not in self.jobs_by_priority:
            self.jobs_by_priority[priority] = deque()
            heapq.heappush(self.priorities_heap, -priority)
        self.jobs_by_priority[priority].append((job_id, data))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        data = self._job_data.get(job_id)
        if data is None:
            raise ValueError(f"Job {job_id} not found")
        success, attempts = self.retry_policy.run(processor, data)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        while self.priorities_heap:
            highest_neg_prio = self.priorities_heap[0]
            highest_prio = -highest_neg_prio
            
            dq = self.jobs_by_priority.get(highest_prio)
            if dq and len(dq) > 0:
                job_id, data = dq.popleft()
                if not dq:
                    heapq.heappop(self.priorities_heap)
                return (job_id, data)
            else:
                heapq.heappop(self.priorities_heap)
        return None