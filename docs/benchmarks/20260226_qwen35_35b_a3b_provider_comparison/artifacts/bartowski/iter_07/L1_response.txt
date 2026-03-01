from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"id": job_id, "data": data})
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def process(self) -> Optional[dict]:
        if self._queue:
            job = self._queue.popleft()
            result = job["data"]
            self._results[job["id"]] = result
            return result
        return None