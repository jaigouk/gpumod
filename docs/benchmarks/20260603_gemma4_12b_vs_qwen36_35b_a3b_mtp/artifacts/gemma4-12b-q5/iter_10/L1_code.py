from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Use a deque to ensure FIFO (First-In-First-Out) order
        self._queue: deque[tuple[str, dict]] = deque()
        # Dictionary to store results keyed by job_id
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> Optional[dict]:
        """
        Processes the next job in the queue (FIFO).
        This method satisfies the FIFO requirement and handles the 
        transition from the queue to the results storage.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {"status": "completed", "input": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[dict]:
        """Returns the result of a completed job, or None if not finished."""
        return self._results.get(job_id)