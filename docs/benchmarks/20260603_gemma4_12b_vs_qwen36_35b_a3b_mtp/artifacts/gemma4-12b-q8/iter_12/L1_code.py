from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.queue: deque = deque()
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self.queue.append({"id": job_id, "data": data})
        return job_id

    def process_jobs(self) -> None:
        """Processes all jobs in the queue in FIFO order."""
        while self.queue:
            job = self.queue.popleft()
            job_id = job["id"]
            data = job["data"]

            # Simulate job processing logic
            self.results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self.results.get(job_id)