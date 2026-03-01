from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, Optional[Dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"id": job_id, "data": data})
        self._results[job_id] = None
        return job_id

    def process(self):
        while self._queue:
            job = self._queue.popleft()
            # Simulate job processing
            self._results[job["id"]] = {"status": "completed", "data": job["data"]}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)