from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self._queue = deque()
            self._jobs = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self._jobs[job_id] = {
                "data": data,
                "result": None,
                "status": "pending"
            }
            self._queue.append(job_id)
            return job_id

        def process_next_job(self, worker_func):
            """Helper to simulate processing."""
            if not self._queue:
                return
            
            job_id = self._queue.popleft()
            job = self._jobs[job_id]
            result = worker_func(job["data"])
            job["result"] = result
            job["status"] = "completed"

        def get_result(self, job_id: str) -> Optional[dict]:
            job = self._jobs.get(job_id)
            if job and job["status"] == "completed":
                return job["result"]
            return None