import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self) -> None:
        self._queue: collections.deque[str] = collections.deque()
        self._jobs: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: Dict) -> str:
        """Add a new job to the end of the queue and return its ID."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """
        Retrieve the result of a completed job.

        The method respects FIFO order: the requested job must be the
        first element in the internal queue. If it is not, a ValueError
        is raised.
        """
        if job_id not in self._jobs:
            return None  # Job already processed or not present

        # Enforce FIFO ordering
        if self._queue[0] != job_id:
            raise ValueError(f"Cannot get result for {job_id} out of FIFO order "
                             f"(next job is {self._queue[0]})")

        result = self._jobs[job_id]
        # Remove the job from both the dict and the queue
        del self._jobs[job_id]
        self._queue.popleft()
        return result