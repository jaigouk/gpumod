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

    def run(self, fn: Callable, data) -> tuple[bool, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                fn(data)
                return True, attempt
            except Exception:
                pass
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs: list[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority, retries=0))

    def process_job(self, job_id: str, processor: Callable) -> bool:
        for job in self._jobs:
            if job.id == job_id:
                policy = RetryPolicy()
                success, _ = policy.run(processor, job.data)
                if not success:
                    job.retries += 1
                return success
        return False

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None
        best_idx = 0
        for i in range(1, len(self._jobs)):
            if self._jobs[i].priority > self._jobs[best_idx].priority:
                best_idx = i
        job = self._jobs.pop(best_idx)
        return job.id, job.data