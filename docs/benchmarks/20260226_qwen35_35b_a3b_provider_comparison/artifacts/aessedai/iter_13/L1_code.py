from collections import deque
from typing import Optional


class JobQueue:
    def __init__(self):
        self._job_queue = deque()
        self._job_results = {}
        self._job_status = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return the job_id."""
        self._job_queue.append(job_id)
        self._job_results[job_id] = None
        self._job_status[job_id] = "pending"
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        """Get the result of a completed job."""
        if job_id not in self._job_results:
            return None
        if self._job_status.get(job_id) != "completed":
            return None
        return self._job_results[job_id]

    def process_next(self) -> Optional[str]:
        """Process the next job in FIFO order and return its job_id if successful."""
        if not self._job_queue:
            return None
        job_id = self._job_queue.popleft()
        if job_id in self._job_status and self._job_status[job_id] == "pending":
            self._job_status[job_id] = "completed"
            self._job_results[job_id] = {"status": "success", "data": self._job_results[job_id]}
            return job_id
        return None

    def process_all(self) -> list:
        """Process all pending jobs in FIFO order."""
        processed = []
        while True:
            job_id = self.process_next()
            if job_id is None:
                break
            processed.append(job_id)
        return processed

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Return the status of a job: 'pending', 'completed', or None if not found."""
        return self._job_status.get(job_id)