from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self.pending_jobs: deque = deque()
        self.jobs_data: Dict[str, dict] = {}
        self.results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append(job_id)
        self.jobs_data[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id)

    def get_next_job(self) -> tuple[str, dict] | None:
        if self.pending_jobs:
            job_id = self.pending_jobs.popleft()
            data = self.jobs_data.get(job_id)
            return job_id, data
        return None

    def set_result(self, job_id: str, result: dict) -> None:
        self.results[job_id] = result