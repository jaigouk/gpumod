from dataclasses import dataclass
from typing import Callable

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

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self.jobs.append(new_job)

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        job = next((j for j in self.jobs if j.id == job_id), None)
        if job is None:
            return False

        policy = RetryPolicy()
        success, attempts = policy.run(Processor, job.data)
        job.retries = attempts
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self.jobs:
            return None

        # Python's sort is stable, so sorting by priority descending 
        # maintains the FIFO order for jobs with the same priority.
        self.jobs.sort(key=lambda x: x.priority, reverse=True)
        job = self.jobs.pop(0)
        return (job.id, job.data)