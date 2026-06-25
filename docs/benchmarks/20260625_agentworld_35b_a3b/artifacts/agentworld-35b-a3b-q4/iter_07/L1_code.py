from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._pending_jobs: deque = deque()
        self._job_data: Dict[str, dict] = {}
        self._job_results: Dict[str, Optional[dict]] = {}
        self._job_completed: Dict[str, bool] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._job_data:
            raise ValueError(f"Job {job_id} already exists")
        self._job_data[job_id] = data
        self._job_completed[job_id] = False
        self._pending_jobs.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        if self._job_completed.get(job_id):
            return self._job_results.get(job_id)
        return None

    def complete_job(self, job_id: str, result: dict) -> None:
        if job_id not in self._job_data:
            raise ValueError(f"Job {job_id} does not exist")
        if self._job_completed.get(job_id):
            raise ValueError(f"Job {job_id} is already completed")
        self._job_completed[job_id] = True
        self._job_results[job_id] = result
        if job_id in self._pending_jobs:
            self._pending_jobs.remove(job_id)