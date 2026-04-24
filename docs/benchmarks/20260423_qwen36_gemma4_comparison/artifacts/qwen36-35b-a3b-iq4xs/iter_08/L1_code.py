from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"job_id": job_id, "data": data})
        return job_id

    def process(self) -> Optional[dict]:
        if self._queue:
            job = self._queue.popleft()
            self._results[job["job_id"]] = {"status": "completed", "data": job["data"]}
            return self._results[job["job_id"]]
        return None

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)