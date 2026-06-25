from collections import deque
from typing import Any, Callable, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self, task_func: Callable[[dict], dict]) -> None:
        """
        Processes the next job in the queue (FIFO) using the provided task function.
        """
        if not self._queue:
            return

        job_id, data = self._queue.popleft()
        try:
            result = task_func(data)
            self._results[job_id] = result
        except Exception as e:
            self._results[job_id] = {"error": str(e)}

    def get_result(self, job_id: str) -> Optional[dict]:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)