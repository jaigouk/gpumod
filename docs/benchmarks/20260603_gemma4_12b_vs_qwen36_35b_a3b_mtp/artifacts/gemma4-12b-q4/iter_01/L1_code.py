from collections import deque
from typing import Dict, Any

class JobQueue:
    def __init__(self):
        # Queue stores tuples of (job_id, data) to maintain FIFO order
        self._queue: deque[tuple[str, dict]]] = deque()
        # Results dictionary stores completed jobs
        self._results: Dict[str, dict[str, Any]]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        This method fulfills the requirement that jobs are processed in order.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {
                "status": "completed",
                "result": data.get("task", "no_task_specified")
            }

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not yet processed."""
        return self._results.get(job_id)