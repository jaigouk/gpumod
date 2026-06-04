from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs as (job_id, data) tuples to maintain FIFO order
        self.queue = deque()
        # Stores results mapped by job_id
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and returns the unique job_id.
        """
        self.queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job, or None if not completed.
        """
        return self.results.get(job_id)

    def process_next_job(self) -> dict | None:
        """
        Processes the next job in the queue (FIFO) and stores its result.
        This method ensures the FIFO requirement is met.
        """
        if not self.queue:
            return None

        job_id, data = self.queue.popleft()

        # Simulate processing logic
        result = {
            "job_id": job_id,
            "status": "completed",
            "original_data": data
        }

        self.results[job_id] = result
        return result