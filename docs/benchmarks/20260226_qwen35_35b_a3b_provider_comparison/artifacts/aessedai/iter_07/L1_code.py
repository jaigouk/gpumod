from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._results:
            return job_id  # Already processed
        
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def _process_job(self, job_id: str):
        if job_id in self._jobs and job_id not in self._results:
            data = self._jobs[job_id]
            self._results[job_id] = {"status": "completed", "data": data}
            del self._jobs[job_id]

    def process(self):
        while self._queue:
            job_id = self._queue.popleft()
            self._process_job(job_id)