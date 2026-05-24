import uuid
from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)