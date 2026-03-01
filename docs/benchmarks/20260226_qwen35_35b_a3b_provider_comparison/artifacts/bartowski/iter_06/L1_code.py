from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._results or any(job['job_id'] == job_id for job in self._queue):
            raise ValueError(f"Job ID {job_id} already exists")
        self._queue.append({'job_id': job_id, 'data': data})
        return job_id

    def process(self) -> Optional[str]:
        if not self._queue:
            return None
        job = self._queue.popleft()
        job_id = job['job_id']
        data = job['data']
        self._results[job_id] = {"status": "completed", "data": data}
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)