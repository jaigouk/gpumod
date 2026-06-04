from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Internal storage for jobs waiting to be processed (FIFO)
        self.queue = deque()
        # Internal storage for results of completed jobs
        self.results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and returns the job_id.
        """
        self.queue.append((job_id, data))
        self.results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job, or None if not completed.
        """
        return self.results.get(job_id)

    def process_next(self) -> None:
        """
        Processes the next job in the queue (FIFO) and stores the result.
        This method facilitates the 'processed' state required by the prompt.
        """
        if self.queue:
            job_id, data = self.queue.popleft()
            # Simulate job processing logic
            self.results[job_id] = {
                "status": "completed",
                "data": data
            }