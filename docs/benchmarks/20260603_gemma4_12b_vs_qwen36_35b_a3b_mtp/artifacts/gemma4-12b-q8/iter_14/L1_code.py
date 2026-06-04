from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Internal storage for jobs to be processed (FIFO)
        self._queue = deque()
        # Internal storage for completed job results
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue. 
        Returns the unique job_id provided.
        """
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Get the result of a completed job by its job_id.
        Returns None if the job hasn't been completed yet.
        """
        return self._results.get(job_id)

    def process_next_job(self) -> None:
        """
        Helper method to process the next job in FIFO order.
         This satisfies the requirement that jobs are processed in FIFO order.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate job processing
            self._results[job_id] = {"status": "completed", "result": data}