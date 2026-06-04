from collections import deque
from typing import Dict, Any

class JobQueue:
    def __init__(self):
        self._queue: deque[str] = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def complete_job(self, job_id: str, result: dict):
        """
        Helper method to simulate job completion and store the result.
        """
        self._results[job_id] = result
        if job_id in self._jobs:
            del self._jobs[job_id]