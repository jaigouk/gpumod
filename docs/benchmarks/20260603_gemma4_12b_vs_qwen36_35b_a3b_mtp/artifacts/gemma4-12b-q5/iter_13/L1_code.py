from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores the order of job_ids to be processed (FIFO)
        self._queue: deque[str] = deque()
        # Stores the raw data for jobs still in the queue
        self._jobs: Dict[str, dict] = {}
        # Stores results for completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns its unique job_id."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def complete_job(self, job_id: str, result: dict):
        """Marks a job as completed and stores its result. """
        self._results[job_id] = result
        # Remove from pending jobs storage
        if job_id in self._jobs:
            del self._jobs[job_id]

    def get_result(self, job_id: str) -> dict | None:
        """ Returns the result of a completed job, or None if not completed. """
        return self._results.get(job_id)