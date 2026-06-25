from dataclasses import dataclass
from typing import Callable, Tuple, Any, Optional

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
                success = fn(data)
                return True, attempts + 1
            except:
                attempts += 1
        return False, attempts

class JobQueue:
    def __init__(self):
        self._jobs = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs.append(Job(id=job_id, data=data, priority=priority))

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if not self._jobs:
            return None
        max_index = 0
        max_priority = self._jobs[0].priority
        for i, job in enumerate(self._jobs):
            if job.priority > max_priority or (job.priority == max_priority and i < max_index):
                max_priority = job.priority
                max_index = i
        return (self._jobs[max_index].id, self._jobs[max_index].data)

    def process_job(self, job_id: str, processor: Callable[[dict], Any]) -> bool:
        for idx, job in enumerate(self._jobs):
            if job.id == job_id:
                retry = RetryPolicy()
                success, attempts = retry.run(processor, job.data)
                # Optionally store attempts in job.retries if needed
                del self._jobs[idx]
                return success
        raise KeyError(f"Job {job_id} not found")