from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process(self):
        while self._queue:
            job_id = self._queue.popleft()
            if job_id in self._jobs:
                data = self._jobs[job_id]
                # Simulate job processing
                result = {"status": "completed", "data": data}
                self._results[job_id] = result
                del self._jobs[job_id]

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)