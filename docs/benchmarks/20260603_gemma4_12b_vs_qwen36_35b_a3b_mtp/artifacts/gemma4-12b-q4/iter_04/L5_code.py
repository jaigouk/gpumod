from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, List


@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0


class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable[[dict]], data: dict) -> Tuple[bool, int]:
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

    def get_next_job(self) -> Optional[Tuple[str, dict]] :
        if not self.jobs:
            return None

        max_priority = max(job.priority for job in self.jobs)
        for i, job in enumerate(self.jobs):
            if job.priority == max_priority:
                return job.id, job.data
        return None

    def process_job(self, job_id: str, Processor: Callable[[dict]]) -> bool:
        job_index = -1
        for i, job in enumerate(self.jobs):
            if job.id == job_id:
                job_index = i
                break

        if job_index == -1:
            return False

        job = self.jobs[job_index]
        success, attempts = self.retry_policy.run(Processor, job.data)

        # Update retries and remove job from queue
        job.retries = attempts - 1 if not success else job.retries
        self.jobs.pop(job_index)

        return success