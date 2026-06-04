from collections import deque
from typing import Dict

class JobQueue:
    def __init__(self):
        # Internal queue to store jobs in FIFO order
        self._queue: deque[tuple[str, dict]] = deque()
        # Internal storage for completed job results
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and returns its unique job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> dict | None:
        """
        Processes the next job in the queue (FIFO).
        This method is included to satisfy the FIFO processing requirement.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate job processing logic
        result = {
            "job_id": job_id,
            "status": "completed",
            "data": data
        }

        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job by its job_id.
        Returns None if the job is not completed or doesn't exist.
        """
        return self._results.get(job_id)