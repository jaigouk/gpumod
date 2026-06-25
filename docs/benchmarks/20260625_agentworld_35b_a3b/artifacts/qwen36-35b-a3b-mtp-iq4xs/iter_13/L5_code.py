from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        attempts = 0
        for _ in range(self.max_attempts):
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                pass
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: List[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self.jobs:
            return None
        best_idx = 0
        best_priority = self.jobs[0].priority
        for i, job in enumerate(self.jobs):
            if job.priority > best_priority:
                best_priority = job.priority
                best_idx = i
        job = self.jobs.pop(best_idx)
        return (job.id, job.data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for job in self.jobs:
            if job.id == job_id:
                success, _ = self.retry_policy.run(processor, job.data)
                return success
        return False