from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue in FIFO order."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> Optional[str]:
        """
        Processes the next job in the queue (FIFO).
        In a real-world scenario, this would involve executing the task.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {
            "status": "completed",
            "processed_data": data
        }

        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the result of a completed job, or None if not found/not completed."""
        return self._results.get(job_id)