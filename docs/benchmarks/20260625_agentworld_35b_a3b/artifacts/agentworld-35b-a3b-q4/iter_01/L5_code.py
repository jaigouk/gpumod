from dataclasses import dataclass
from typing import Callable, Tuple, Optional

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data) -> Tuple[bool, int]:
        attempts_made = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts_made = attempt
            try:
                fn(data)
                return (True, attempts_made)
            except Exception:
                pass
        return (False, attempts_made)

class JobQueue:
    def __init__(self):
        self._job_entries: list[tuple[Job, int]] = []
        self._job_by_id: dict[str, Job] = {}
        self._counter = 0

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self._job_entries.append((job, self._counter))
        self._job_by_id[job_id] = job
        self._counter += 1

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self._job_entries:
            return None
        
        best_idx = -1
        best_priority = -float('inf')
        best_counter = float('inf')
        
        for i, (job, counter) in enumerate(self._job_entries):
            if job.priority > best_priority:
                best_priority = job.priority
                best_counter = counter
                best_idx = i
            elif job.priority == best_priority:
                if counter < best_counter:
                    best_counter = counter
                    best_idx = i
        
        if best_idx != -1:
            job, _ = self._job_entries.pop(best_idx)
            return (job.id, job.data)
        return None

    def process_job(self, job_id: str, processor: Callable) -> bool:
        retry_policy = RetryPolicy()
        job = self._job_by_id.get(job_id)
        if not job:
            return False
        success, _ = retry_policy.run(processor, job.data)
        return success