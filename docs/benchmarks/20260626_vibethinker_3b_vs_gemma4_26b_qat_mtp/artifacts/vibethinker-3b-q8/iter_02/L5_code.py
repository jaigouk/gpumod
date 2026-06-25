from dataclass import dataclass
from typing import Callable, Tuple, Dict, Optional

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts
        self._attempts = 0

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            try:
                fn(data)
                return True, attempts + 1
            except Exception:
                attempts += 1
        return False, self.max_attempts

class JobQueue:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._policy = RetryPolicy()

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        self._jobs[job_id] = Job(job_id, data, priority, 0)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job id {job_id} not found")
        success, _ = self._policy.run(processor, job.data)
        return success

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        highest_priority = -float('inf')
        selected = None
        for job in self._jobs.values():
            if job.priority > highest_priority:
                highest_priority = job.priority
                selected = job
        return (selected.id, selected.data) if selected else None