from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, Any] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)

    def process_next(self) -> Optional[tuple]:
        return self._queue.popleft() if self._queue else None

    def complete_job(self, job_id: str, result: dict) -> None:
        self._results[job_id] = result