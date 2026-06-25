from collections import deque
from typing import Callable

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)

    def process(self, processor: Callable[[dict], dict]) -> str | None:
        if not self._queue:
            return None
        job_id, data = self._queue.popleft()
        self._results[job_id] = processor(data)
        return job_id