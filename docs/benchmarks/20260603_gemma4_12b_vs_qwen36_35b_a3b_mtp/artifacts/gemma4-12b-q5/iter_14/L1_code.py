from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs as (job_id, data) in FIFO order
        self.queue: deque = deque()
        # Stores results of completed jobs
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return the job_id."""
        self.queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job by its ID."""
        return self.results.get(job_id)

    def process_next_job(self) -> bool:
        """
        Internal helper to simulate processing. 
        Moves a job from the queue to the results dictionary.
        """
        if self.queue:
            job_id, data = self.queue.popleft()
            # Simulate processing logic
            self.results[job_id] = {"status": "completed", "data": data}
            return True
        return False