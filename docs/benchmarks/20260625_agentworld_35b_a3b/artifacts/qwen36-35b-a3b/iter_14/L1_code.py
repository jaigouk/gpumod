from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def process(self) -> None:
        if self._queue:
            job_id, data = self._queue.popleft()
            self._results[job_id] = {"status": "completed", "data": data}