from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._jobs[job_id] = data
        self._results[job_id] = None  # Initialize result
        return job_id

    def process(self):
        if self._queue:
            job_id = self._queue.popleft()
            # Simulate processing
            self._results[job_id] = {"status": "completed", "data": self._jobs[job_id]}

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)