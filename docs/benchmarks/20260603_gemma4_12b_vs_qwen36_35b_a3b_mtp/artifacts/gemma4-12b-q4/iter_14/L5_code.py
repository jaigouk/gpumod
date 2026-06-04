from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional, Union

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
            attempts += 1
            try:
                fn(data)
                return True, attempts
            except Exception:
                continue
        return False, attempts

class JobQueue:
    def __init__(self):
        self._jobs: List[Job] = []
        self._retry_policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority)))

    def process_job(self, job_id: str, Processor: Callable[[dict]]) -> bool:
        job = next((j for j in self._jobs if j.id == job_id), None)
        if job is None:
            return False
        success, _ = self._retry_policy.run(Processor, job.data)
        return success

    def get_next_job(self) -> Union[Tuple[str, dict], None]:
        if not self._jobs:
            return None

        # Find highest priority
        max_priority = max(j.priority for j in self._jobs)

        # Return first occurrence (FIFO) of highest priority
        for j in self._jobs:
            if j.priority == max_priority:
                return (j.id, j.data)
        return None