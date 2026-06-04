from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores job IDs in FIFO order
        self.queue: deque[str] = deque()
        # Stores job data until processed
        self.jobs: Dict[str, dict] = {}
        # Stores results of completed jobs
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return its ID."""
        self.jobs[job_id] = data
        self.queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """Retrieve the result of a completed job by its ID."""
        return self.results.get(job_id)

    def complete_job(self, job_id: str, result: dict):
        """
        Helper method to simulate the processing of a job.
         Completes a job by moving it from jobs to results.
        """
        if job_id in self.jobs:
            self.results[job_id] = result
            del self.jobs[job_id]