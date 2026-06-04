from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        # Store jobs in a deque to ensure FIFO (First-In-First-Out ) order
        self._queue: deque[tuple[str, dict]] = deque()
        # Store results keyed by job_id
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return its job_id."""
        self._queue.append((job_id, data))
        self._results[job_id] = None  # Initialize as not processed
        return job_id

    def process_jobs(self) -> None:
        """
        Processes all jobs in the queue in FIFO order.
        This method is included to satisfy the requirement that jobs 
        are processed and moved from the queue to the results storage.
        """
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job by its job_id."""
        return self._results.get(job_id)