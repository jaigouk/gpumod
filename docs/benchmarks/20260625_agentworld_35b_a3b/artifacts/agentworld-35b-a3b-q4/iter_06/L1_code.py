import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.pending_jobs: collections.deque = collections.deque()
        self.results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id)

    def process_next(self) -> dict | None:
        if not self.pending_jobs:
            return None
        job_id, data = self.pending_jobs.popleft()
        result = {"status": "completed", "data": data}
        self.results[job_id] = result
        return result