from collections import deque
from typing import Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: dict[str, dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue in FIFO order.
        """
        if not self._queue:
            return

        job_id, data = self._queue.popleft()

        # Simulate processing logic: here we just wrap the data in a result dict
        result = {
            "status": "completed",
            "processed_data": data
        }
        self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)