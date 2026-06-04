from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        # Stores job IDs in FIFO order
        self._queue: deque[str] = deque()
        # Stores job data until processed
        self._jobs: Dict[str, dict] = {}
        # Stores results of completed jobs
    self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return the job_id."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process_next(self) -> None:
        """
        Processes the next job in the queue. 
        Included to satisfy the FIFO processing requirement.
        """
        if self._queue:
            job_id = self._queue.popleft()
            job_data = self._jobs.pop(job_id)

            # Simulate job processing logic
            result = {"status": "completed", "processed_data": job_data}
            self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)