from collections import deque
from typing import Dict, Optional


class JobQueue:
    def __init__(self) -> None:
        # FIFO queue holding job identifiers in order of addition
        self._queue: deque[str] = deque()
        # Map from job_id to its data
        self._jobs: dict[str, Dict] = {}
        # Map from job_id to its resulting data (None if not computed yet)
        self._results: dict[str, Optional[Dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a new job to the end of the queue and return its ID."""
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        """
        Get the result of a job after processing it or any earlier jobs
        in FIFO order. Returns the result if the job has been completed,
        otherwise returns None.
        """
        # Process jobs from the front of the queue until we complete the requested job.
        while self._queue:
            current_id = self._queue[0]  # Look at the head
            # Remove the job from the queue (it's about to be processed)
            self._queue.popleft()
            # Compute the result (here we simply return the provided data)
            result = self._jobs[current_id]
            # Record the result for this job
            self._results[current_id] = result
            # If this is the job we were asked about, return its result.
            if current_id == job_id:
                return result
        # If we exit the loop, the queue is empty. The requested job may have
        # been processed earlier, in which case its result is stored in _results.
        return self._results.get(job_id)