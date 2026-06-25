import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.pending_jobs = collections.deque()
        self.job_data: Dict[str, dict] = {}
        self.completed_results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append(job_id)
        self.job_data[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.completed_results.get(job_id)

    def process_next(self) -> dict | None:
        if not self.pending_jobs:
            return None
        job_id = self.pending_jobs.popleft()
        data = self.job_data.get(job_id, {})
        result = {"status": "completed", "data": data}
        self.completed_results[job_id] = result
        return result