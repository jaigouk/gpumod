from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._jobs: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job {job_id} already exists.")
        self._queue.append(job_id)
        self._jobs[job_id] = {
            "data": data,
            "status": "pending",
            "result": None
        }
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        if job_id not in self._jobs:
            return None
        
        job_info = self._jobs[job_id]
        if job_info["status"] == "completed":
            return job_info["result"]
        return None

    def process_job(self, job_id: str, result: dict):
        if job_id in self._jobs and self._jobs[job_id]["status"] == "pending":
            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["result"] = result
            if self._queue and self._queue[0] == job_id:
                self._queue.popleft()