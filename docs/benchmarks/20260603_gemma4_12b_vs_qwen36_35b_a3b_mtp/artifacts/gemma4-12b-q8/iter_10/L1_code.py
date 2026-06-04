from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque[tuple[str, dict]] = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and returns the job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def process_jobs(self) -> None:
        """
        Processes all jobs currently in the queue in FIFO order.
        """
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job or None if not completed.
        """
        return self._results.get(job_id)