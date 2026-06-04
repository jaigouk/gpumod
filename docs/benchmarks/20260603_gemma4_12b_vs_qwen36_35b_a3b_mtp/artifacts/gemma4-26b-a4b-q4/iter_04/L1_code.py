from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._jobs[job_id] = {
            "data": data,
            "result": None,
            "completed": False
        }
        self._queue.append(job_id)
        return job_id

    def process_next_job(self):
        """
        Processes the next job in the queue using FIFO order.
        In a real application, this might involve running a worker function.
        """
        if not self._queue:
            return

        job_id = self._queue.popleft()
        job = self._jobs[job_id]

        # Simulating job processing logic
        # Here we simply return the data back in a result dictionary
        job["result"] = {"status": "success", "output": job["data"]}
        job["completed"] = True

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job and job["completed"]:
            return job["result"]
        return None