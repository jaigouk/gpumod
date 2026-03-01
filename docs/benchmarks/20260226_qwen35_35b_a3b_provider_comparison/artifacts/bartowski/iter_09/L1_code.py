from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._jobs: dict = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job with ID {job_id} already exists")
        self._queue.append(job_id)
        self._jobs[job_id] = {
            "data": data,
            "status": "pending",
            "result": None
        }
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job["status"] == "completed":
                return job["result"]
        return None

    def process_jobs(self):
        while self._queue:
            job_id = self._queue.popleft()
            if job_id in self._jobs and self._jobs[job_id]["status"] == "pending":
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["result"] = {
                    "status": "completed",
                    "data": job["data"]
                }