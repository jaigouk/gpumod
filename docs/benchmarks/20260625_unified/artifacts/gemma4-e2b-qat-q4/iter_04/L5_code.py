from dataclasses import dataclass
from typing import Callable, Any, Tuple, List

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: dict) -> tuple[bool, int]:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                result = fn(data)
                return True, attempts
            except Exception as e:
                # In a real scenario, we might log the error. Here we just catch and retry.
                pass
        return False, attempts

class JobQueue:
    def __init__(self, retry_policy: RetryPolicy = None):
        self._jobs: List[Job] = []
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy(4)

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self._jobs.append(new_job)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        # 1. Find the job
        job_to_process = next((job for job in self._jobs if job.id == job_id), None)

        if not job_to_process:
            return False

        # 2. Apply RetryPolicy
        success, attempts_made = self.retry_policy.run(processor, job_to_process.data)

        if success:
            # Simulate successful processing by updating job state
            job_to_process.retries += attempts_made
            return True
        else:
            # In a real system, we might move the job to a dead letter queue
            return False

    def get_next_job(self) -> Tuple[str, dict] | None:
        if not self._jobs:
            return None

        # Sort criteria: 1. Priority (descending), 2. FIFO (implicit by list order for ties)
        # We sort by priority descending. We don't need a secondary key for FIFO 
        # if we maintain insertion order for equal priorities, which Python's sort is stable on.
        sorted_jobs = sorted(self._jobs, key=lambda job: job.priority, reverse=True)

        if sorted_jobs:
            return (sorted_jobs[0].id, sorted_jobs[0].data)

        return None