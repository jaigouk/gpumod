from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional, List

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        self.jobs: List[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority)))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        idx = -1
        for i, job in enumerate(self.jobs):
            if job.id == job_id:
                idx = i
                break

        if idx == -1:
            return False

        job = self.jobs[idx]
        success, _ = self.retry_policy.run(processor, job.data)
        self.jobs.pop(idx)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        max_priority = max(j.priority for j in self.jobs)
        for i, job in enumerate(self.jobs):
            if job.priority == max_priority:
                result = (job.id, job.data)
                self.jobs.pop(i)
                return result
        return None