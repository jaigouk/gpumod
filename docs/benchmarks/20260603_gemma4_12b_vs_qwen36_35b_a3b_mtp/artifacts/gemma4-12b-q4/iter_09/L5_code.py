from dataclasses import dataclass
from typing import Callable
from __future__ import annotations

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
        while attemps < self.max_attempts:
            try:
                fn(data)
                return True, attemps + 1
            except Exception:
                attempts += 1
        return False, attemps

class JobQueue:
    def __init__(self):
        self._jobs: list[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority)))

    def process_job(self, job_id: str, Processor: Callable) -> bool:
        idx = -1
        for i, job in enumerate(self._jobs):
            if job.id == job_id:
                idx = i
                break

        if idx == -1:
            return False

        job = self._jobs[idx]
        policy = RetryPolicy()
        success, _ = policy.run(Processor, job.data)
        self._jobs.pop(idx)
        return success

    def get_next_job(self) -> tuple[str, dict] | None:
        if not self._jobs:
            return None

        best_idx = 0
        for i in range(1, len(self._jobs)):
            if self._jobs[i].priority > self._jobs[best_idx]].priority:
                best_idx = i

        return (self._jobs[best_idx]].id, self._jobs[best_idx]].data))