from collections import deque
from typing import Dict, Optional, Tuple, Deque


class JobQueue:
    def __init__(self) -> None:
        # Store jobs in FIFO order as a deque of (job_id, data) tuples
        self._queue: Deque[Tuple[str, Dict]] = deque()

    def add_job(self, job_id: str, data: Dict) -> str:
        """Add a new job to the end of the queue and return its job_id."""
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """
        Retrieve the result of a completed job.
        Returns the stored data dict for the given job_id, or None if not found.
        """
        for i, (current_job_id, current_data) in enumerate(self._queue):
            if current_job_id == job_id:
                # Mark the job as completed by removing it from the queue
                del self._queue[i]
                return current_data
        # Job not found in the queue (either never added or already completed)
        return None