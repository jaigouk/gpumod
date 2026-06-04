from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores the IDs of jobs in FIFO order
        self._queue: deque[str] = deque()
        # Stores the raw data for jobs currently in the queue
        self._jobs: Dict[str, dict] = {}
        # Stores the results of processed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        Returns the job_id.
        """
        if job_id in self._jobs or job_id in self._results:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Retrieves the result of a completed job by its ID.
        Returns None if the job is not completed.
        """
        return self._results.get(job_id)

    def process_next_job(self) -> Optional[dict]:
        """
        Processes the next job in the queue (FIFO).
        This method is provided to satisfy the 'processed' requirement.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Simulate processing logic
        result = {"status": "success", "processed_data": data}
        self._results[job_id] = result
        return result