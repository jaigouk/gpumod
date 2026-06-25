from __future__ import annotations
import heapq
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, List, Set

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                continue
        return False, self.max_attempts

class JobQueue:
    def __init__(self) -> None:
        self.heap: List[Tuple[int, int, Job]] = []
        self.job_dict: Dict[str, Job] = {}
        self.counter: int = 0
        self.removed: Set[str] = set()
        selfretry_policy = RetryPolicy(max_attempts=4)

    def add_job(self, job_id: str, data: Dict[str, Any], priority: int = 0) -> None:
        job = Job(job_id, data, priority, 0)
        self.job_dict[job_id] = job
        heapq.heappush(self.heap, (-priority, self.counter, job))
        self.counter += 1

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.job_dict.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        self.removed.add(job_id)
        del self.job_dict[job_id]

        success, _ = selfretry_policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Tuple[str, Dict[str, Any]] | None:
        while self.heap:
            _, _, job = heapq.heappop(self.heap)
            if job.id in self.removed:
                self.removed.discard(job.id)
                continue
            self.removed.discard(job.id)
            del self.job_dict[job.id]
            return job.id, job.data
        return None