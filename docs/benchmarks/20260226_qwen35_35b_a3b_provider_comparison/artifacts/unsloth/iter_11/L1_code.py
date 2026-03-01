from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, dict] = {}
        self._job_ids: set = set()

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._job_ids:
            raise ValueError(f"Job ID {job_id} is already in the queue or completed.")
        self._queue.append((job_id, data))
        self._job_ids.add(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def process(self):
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate job processing
            self._results[job_id] = {"status": "completed", "data": data}
            self._job_ids.discard(job_id)