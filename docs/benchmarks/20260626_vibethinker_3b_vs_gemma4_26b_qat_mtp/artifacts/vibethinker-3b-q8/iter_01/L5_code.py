from dataclasses import dataclass
from collections import deque
from typing import Callable, Any, Dict, Deque, List, Tuple, Optional

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
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except:
                pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: Dict[int, Deque[Job]] = {}
        self.job_map: Dict[str, Job] = {}
        self._retry_policy = RetryPolicy()
        self._max_priority: Optional[int] = None

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        if priority not in self.jobs:
            self.jobs[priority] = deque()
            if self._max_priority is None or priority > self._max_priority:
                self._max_priority = priority
        job = Job(job_id, data, priority, 0)
        self_jobs = self.jobs[priority]
        self_jobs.append(job)
        self.job_map[job_id] = job

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self.job_map.get(job_id)
        if job is None:
            return False
        priority = job.priority
        self.jobs[priority].remove(job)
        del self.job_map[job_id]
        success, attempts = self._retry_policy.run(processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if self._max_priority is None:
            return None
        dq = self.jobs.get(self._max_priority)
        if not dq:
            self._max_priority = None
            return None
        job = dq.popleft()
        self.job_map.pop(job.id, None)

        if not dq:
            non_empty = [p for p in self.jobs if self.jobs[p]]
            if non_empty:
                self._max_priority = max(non_empty)
            else:
                self._max_priority = None
        return (job.id, job.data)