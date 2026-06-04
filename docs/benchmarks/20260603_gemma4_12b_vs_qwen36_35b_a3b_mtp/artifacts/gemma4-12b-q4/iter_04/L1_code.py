from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # FIFO queue to store jobs as (job_id, data)
        self._queue: deque[tuple[str, dict]]] = deque()
        # Dictionary to store results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue and return the job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the FIFO queue.
        This method fulfills the requirement of processing jobs in order.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "original_data": data}

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a completed job. Returns None if not completed.
        """
        return self._results.get(job_id)