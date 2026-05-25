from collections import deque
from typing import Optional, Dict

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)