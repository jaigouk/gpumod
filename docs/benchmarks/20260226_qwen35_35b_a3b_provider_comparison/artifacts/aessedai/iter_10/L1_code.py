from typing import Dict, Optional
from collections import deque


class JobQueue:
    def __init__(self):
        self._job_queue: deque = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            return job_id
        self._jobs[job_id] = data
        self._job_queue.append(job_id)
        return job_id

    def process(self) -> None:
        while self._job_queue:
            job_id = self._job_queue.popleft()
            if job_id in self._jobs:
                data = self._jobs.pop(job_id)
                self._results[job_id] = {"status": "completed", "input": data}

    def get_result(self, job_id: str) -> Optional[dict]:
        if job_id in self._results:
            return self._results[job_id]
        return None