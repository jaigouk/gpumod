from collections import deque
from typing import Any, Dict, Optional

class JobQueue:
    def __init__(self) -> None:
        self._queue = deque()
        self._pending_jobs: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._pending_jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process_next_job(self) -> Optional[Dict[str, Any]]:
        """Processes the next job in the queue in FIFO order."""
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._pending_jobs.pop(job_id)

        # Simulate processing logic
        result = {"status": "success", "processed_data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)