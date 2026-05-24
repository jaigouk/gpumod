from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        self._queue = []
        self._results = {}
        self._status = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"job_id": job_id, "data": data})
        self._status[job_id] = "pending"
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        if job_id in self._results:
            return self._results[job_id]
        return None

    def process(self) -> bool:
        if not self._queue:
            return False
        job = self._queue.pop(0)
        job_id = job["job_id"]
        # Simulate processing
        self._results[job_id] = {"status": "completed", "data": job["data"]}
        self._status[job_id] = "completed"
        return True