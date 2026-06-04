from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque[str] = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the FIFO queue.
        This moves the job from the internal queue to the results dictionary.
        """
        if self._queue:
            job_id = self._queue.popleft()
            job_data = self._jobs.pop(job_id)
            # Simulate processing logic
            self._results[job_id] = {
                "status": "completed",
                "data": job_data
            }

    def get_result(self, job_id: str) -> Optional[dict]]:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)