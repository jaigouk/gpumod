from collections import deque
from typing import Dict

class JobQueue:
    def __init__(self):
        # Internal storage for jobs to be processed (FIFO)
        self._queue: deque[tuple[str, dict]] = deque()
        # Internal storage for results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return its unique ID."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        While not explicitly requested as a method in the list, 
        it is required to satisfy requirements 2 and 3.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job by its ID."""
        return self._results.get(job_id)