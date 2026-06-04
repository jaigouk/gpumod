from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs in FIFO order: (job_id, data)
        self.queue = deque()
        # Stores results of completed jobs: {job_id: result}
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self.queue.append((job_id, data))
        return job_id

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Helper method to simulate the completion of a job.
         This allows get_result to return a value.
        """
        self.results[job_id] = result

    def get_result(self, job_id: str) -> Optional[dict]:
        """Returns the result of a completed job, or None if not completed."""
        return self.results.get(job_id)