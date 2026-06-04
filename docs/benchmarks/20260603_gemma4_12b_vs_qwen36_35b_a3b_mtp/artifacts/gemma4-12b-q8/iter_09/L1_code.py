from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs in FIFO order until they are processed
        self._queue: deque = deque()
        # Stores the results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the unique job_id."""
        self._queue.append({"id": job_id, "data": data})
        return job_id

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Helper method to simulate completion of a job. 
         This allows get_result to retrieve data.
        """
        self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)