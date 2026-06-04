from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> Tuple[bool, int]:
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
        new_job = Job(id=job_id, data=data, priority=priority)
        self.jobs.append(new_job)
        # Python's sort is stable, ensuring FIFO for same priority
        self.jobs.sort(key=lambda x: x.priority, reverse=True)

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        index = -1
        for i, job in enumerate(self.jobs):
            if job.id == job_id:
                index = i
                break

        if index == -1:
            return False

        job = self.jobs[index]
        success, _ = self.retry_policy.run(Processor, job.data)

        if success:
            self.jobs.pop(index)

        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self.jobs:
            return None
        # Peek at the highest-priority job (FIFO for ties)
        job = self.jobs[0]
        return (job.id, job.data)