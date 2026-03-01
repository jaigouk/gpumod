from collections import deque
from typing import Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: dict = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = {"status": "pending", "data": data, "result": None}
        self._queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        if job_id not in self._jobs:
            return None
        job = self._jobs[job_id]
        if job["status"] == "completed":
            return job["result"]
        return None

    def process(self):
        while self._queue:
            job_id = self._queue.popleft()
            if job_id in self._jobs and self._jobs[job_id]["status"] == "pending":
                job = self._jobs[job_id]
                job["status"] = "completed"
                job["result"] = {"processed": True, "data": job["data"]}