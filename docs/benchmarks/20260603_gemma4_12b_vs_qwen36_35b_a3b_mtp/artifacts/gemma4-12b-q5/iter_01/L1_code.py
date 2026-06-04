from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.queue: deque[str] = deque()
        self.jobs: Dict[str, dict] = {}
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.jobs[job_id] = data
        self.queue.append(job_id)
        return job_id

    def process_jobs(self):
        """
        Processes the next job in the FIFO queue.
        This method is included to satisfy the 'processed' requirement.
        """
        if self.queue:
            job_id = self.queue.popleft()
            data = self.jobs.pop(job_id)

            # Simulate processing logic
            result = {"status": "completed", "input_received": data}
            self.results[job_id] = result

    def get_result(self, job_id: str) -> Optional[dict]:
        return self.results.get(job_id)