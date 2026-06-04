from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores jobs in FIFO order: (job_id, data)
        self._queue: deque = deque()
        # Stores completed results: {job_id: result_dict}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue and returns the job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def process_jobs(self) -> None:
        """
        Processes all jobs currently in the queue in FIFO order.
        This method is included to satisfy the requirement that jobs 
        are processed before results can be retrieved.
        """
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Returns the result of a completed job, or None if not completed.
        """
        return self._results.get(job_id)