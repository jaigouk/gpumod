from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Using a deque to maintain FIFO order for the queue
        self._queue: deque[tuple[str, dict]] = deque()
        # Dictionary to store results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job or None if not completed."""
        return self._results.get(job_id)

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        This method is provided to satisfy the 'processed' requirement 
        so that get_result can actually return a value.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate job processing logic here
            self._results[job_id] = {"status": "completed", "result": data}