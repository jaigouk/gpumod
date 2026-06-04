from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List

@dataclass
class Job:
    id: str
    data: Dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
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
        self.jobs: List[Job] = []
        self.retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: Dict, priority: int = 0) -> None:
        self.jobs.append(Job(id=job_id, data=data, priority=priority)))

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False

        success, attempts = self.retry_policy.run(Processor, job.data)
        job.retries += attempts
        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.jobs:
            return None

        max_priority = max(j.priority for j in self.jobs)

        for i, job in enumerate(self.jobs):
            if job.priority == max_priority:
                result = (job.id, job.data)
                self.jobs.pop(i)
                return result
        return None