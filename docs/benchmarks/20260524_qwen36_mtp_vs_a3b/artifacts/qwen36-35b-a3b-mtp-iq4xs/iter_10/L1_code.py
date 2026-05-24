import uuid
from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id not in self._jobs:
            self._jobs[job_id] = data
            self._queue.append(job_id)
        return job_id

    def process(self) -> Optional[dict]:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        # Simulate processing
        result = {"status": "completed", "data": self._jobs.get(job_id)}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)