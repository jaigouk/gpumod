from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Tuple, Set, Optional

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
        self._delay_counter = 0

    def run(self, fn: Callable[[], Any], data: dict) -> tuple[bool, int]:
        attempts = 0
        success = False
        while attempts < self.max_attempts:
            try:
                fn(data)
                success = True
                self._delay_counter += 1
                break
            except Exception as _:
                self._delay_counter += 1
            attempts += 1
        return success, attempts

class JobQueue:
    def __init__(self, max_attempts: int = 4):
        self._max_attempts = max_attempts
        self._retry_policy = RetryPolicy(self._max_attempts)
        self._jobs_by_id: Dict[str, Job] = {}
        self._priority_queues: Dict[int, List[Job]] = {}

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        job = Job(id=job_id, data=data, priority=priority, retries=0)
        self._jobs_by_id[job_id] = job
        self._priority_queues.setdefault(priority, []).append(job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        job = self._jobs_by_id.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        success, attempts = self._retry_policy.run(processor, job.data)
        if job in self._priority_queues.get(job.priority, []):
            self._priority_queues[job.priority].remove(job)
        return success

    def get_next_job(self) -> Tuple[str, dict] | None:
        for priority in sorted(self._priority_queues.keys()):
            queue = self._priority_queues[priority]
            if queue:
                job = queue.pop(0)
                del self._jobs_by_id[job.id]
                return job.id, job.data
        return None