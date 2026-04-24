from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: dict = {}
        self._results: dict = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)

    def process(self) -> str | None:
        if self._queue:
            job_id = self._queue.popleft()
            self._results[job_id] = {"status": "completed", "data": self._jobs.get(job_id)}
            return job_id
        return None