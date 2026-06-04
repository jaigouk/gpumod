from collections import deque
from typing import Dict

class JobQueue:
    def __init__(self):
        # Internal storage for jobs waiting to be processed
        self._queue = deque()
        # Internal storage for completed job results
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """Process the next job in the FIFO queue."""
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)