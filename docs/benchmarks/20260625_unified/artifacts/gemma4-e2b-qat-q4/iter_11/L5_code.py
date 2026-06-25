from dataclasses import dataclass
from typing import Callable, Any, Optional

@dataclass
class Job:
    id: str
    data: dict
    priority: int = 0
    retries: int = 0

class RetryPolicy:
    def __init__(self, max_attempts: int = 4):
        self.max_attempts = max_attempts

    def run(self, fn: Callable, data: Any) -> tuple[bool, int]:
        """
        Executes fn with data, retrying on exception up to max_attempts.
        Returns (success, attempts_made).
        """
        for attempt in range(self.max_attempts):
            try:
                # Note: fn is expected to accept data and return the result
                fn(data)
                return (True, attempt + 1)
            except Exception:
                # Retry logic
                continue

        return (False, self.max_attempts)

class JobQueue:
    def __init__(self):
        self._jobs: list[Job] = []

    def add_job(self, job_id: str, data: dict, priority: int = 0) -> None:
        new_job = Job(id=job_id, data=data, priority=priority)
        self._jobs.append(new_job)

    def get_next_job(self) -> tuple[str, dict] | None:
        """
        Returns the highest-priority job's (id, data); FIFO order within the same priority.
        """
        if not self._jobs:
            return None

        best_job_index = -1
        max_priority = -float('inf')

        # Find the job with the highest priority. 
        # FIFO within the same priority is handled by iterating sequentially 
        # and only updating when priority is strictly greater.
        for i, job in enumerate(self._jobs):
            if job.priority > max_priority:
                max_priority = job.priority
                best_job_index = i

        if best_job_index == -1:
            return None

        job = self._jobs[best_job_index]
        return (job.id, job.data)

    def process_job(self, job_id: str, processor: Callable) -> bool:
        """
        Processes a specific job using the RetryPolicy.

        :param job_id: The ID of the job to process.
        :param processor: A callable that takes the job's data as input.
        :return: True if the job succeeded, False otherwise.
        """
        # 1. Find the job
        job_to_process = next((job for job in self._jobs if job.id == job_id), None)

        if not job_to_process:
            return False

        # 2. Instantiate and run the RetryPolicy
        policy = RetryPolicy(max_attempts=4)
        success, attempts_made = policy.run(processor, job_to_process.data)

        # Optional: Update job state if necessary, though not required by prompt
        # job_to_process.retries += (attempts_made - 1) 

        return success