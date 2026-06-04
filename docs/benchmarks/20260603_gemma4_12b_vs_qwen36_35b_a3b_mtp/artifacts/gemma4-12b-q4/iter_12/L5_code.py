from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any

@dataclass
class Job:
    id: str
    data: Dict[str, Any]
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
        attempts_made = 0
        while attempts_made < self.max_attempts:
            try:
                fn(data)
                return True, attempts_made + 1
            except Exception:
                attempts_made += 1
        return False, attempts_made

class JobQueue:
    def __init__(self):
        self.jobs: list[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self.jobs.append(new_job)
        # Python's sort is stable, preserving FIFO for identical priorities
        self.jobs.sort(key=lambda x: x.priority, reverse=True)

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        for i, job in enumerate(self.jobs):
            if job.id == job_id:
                success, _ = self.retry_policy.run(Processor, job.data)
                if success:
                    self.jobs.pop(i)
                return success
        return False

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.jobs:
            return None
        # Returns the highest priority job (first in the sorted list )
        return self.jobs[0].id, self.jobs[0].data