from collections import deque
from typing import Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue in FIFO order.
        Simulates processing by returning the input data in a result dictionary.
        """
        if not self._queue:
            return

        job_id, data = self._queue.popleft()
        # Simulate job processing
        self._results[job_id] = {
            "status": "completed",
            "output": data
        }

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)