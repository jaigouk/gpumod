import collections
from typing import Dict, Optional, Tuple

class JobQueue:
    def __init__(self):
        self._queue: collections.deque = collections.deque()
        self._results: Dict[str, dict | None] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        if self._queue:
            return self._queue.popleft()
        return None

    def set_result(self, job_id: str, result: dict) -> None:
        if job_id in self._results:
            self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)