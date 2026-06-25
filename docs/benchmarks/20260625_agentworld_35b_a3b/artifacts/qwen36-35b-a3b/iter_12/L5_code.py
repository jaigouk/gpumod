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
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        if job_id not in self._jobs:
            self._jobs[job_id] = Job(id=job_id, data=data, priority=priority)
            self._order.append(job_id)

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._order:
            return None
        best_id = max(self._order, key=lambda jid: (self._jobs[jid].priority, -self._order.index(jid)))
        self._order.remove(best_id)
        job = self._jobs.pop(best_id)
        return job.id, job.data

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self._jobs:
            return False
        policy = RetryPolicy()
        success, _ = policy.run(processor, self._jobs[job_id].data)
        return success