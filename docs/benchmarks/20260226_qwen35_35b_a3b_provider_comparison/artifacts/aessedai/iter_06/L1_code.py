from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def process(self) -> None:
        while self._queue:
            job_id, data = self._queue.popleft()
            self._results[job_id] = {"status": "completed", "data": data}