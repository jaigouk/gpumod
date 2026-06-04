from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Internal storage for jobs to be processed (FIFO)
        self._queue = deque()
        # Internal storage for completed job results
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the unique job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        This method is necessary to move jobs from the queue to the results.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job or None if not completed."""
        return self._results.get(job_id)