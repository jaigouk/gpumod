import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.pending_jobs: collections.deque = collections.deque()
        self.completed_results: Dict[str, dict | None] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append({'job_id': job_id, 'data': data})
        return job_id

    def get_next_job(self) -> Optional[tuple[str, dict]]:
        if self.pending_jobs:
            job = self.pending_jobs.popleft()
            return job['job_id'], job['data']
        return None

    def set_result(self, job_id: str, result: dict) -> None:
        self.completed_results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self.completed_results.get(job_id)