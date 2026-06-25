from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def process(self, job_id: str, result: dict) -> None:
        if job_id in self._results:
            self._results[job_id] = result