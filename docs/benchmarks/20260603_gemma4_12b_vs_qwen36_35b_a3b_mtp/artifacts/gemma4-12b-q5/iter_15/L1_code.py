from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores job IDs in FIFO order
        self._queue: deque[str] = deque()
        # Stores input data for pending jobs
        self._jobs: Dict[str, dict] = {}
        # Stores results for completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs or job_id in self._results:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def process_next(self) -> None:
        """
        Helper method to process the next job in the FIFO queue.
         This satisfies the requirement that jobs are processed in FIFO order.
        """
        if self._queue:
            job_id = self._queue.popleft()
            # Simulate processing logic
            data = self._jobs.pop(job_id, {})
            self._results[job_id] = {
                "status": "completed",
                "original_data": data,
                "result": "success"
            }