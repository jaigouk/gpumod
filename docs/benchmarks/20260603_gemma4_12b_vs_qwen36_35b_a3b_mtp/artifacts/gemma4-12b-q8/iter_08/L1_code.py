from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)

    def complete_job(self, job_id: str, result: dict) -> None:
        """Helper method to set the result of a processed job. """
        if job_id in self._results:
            self._results[job_id] = result