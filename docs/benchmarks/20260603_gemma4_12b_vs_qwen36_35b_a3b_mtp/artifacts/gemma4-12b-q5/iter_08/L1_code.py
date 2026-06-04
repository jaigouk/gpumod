from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque[str] = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs or job_id in self._results:
            raise ValueError("Job ID must be unique")

        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process_next(self) -> Optional[str]:
        """
        Processes the next job in the queue (FIFO).
        This method is provided to satisfy the requirement that jobs 
        be processed in FIFO order.
        """
        if not self._queue:
            return None

        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)

        # Simulate job processing logic
        result = {
            "status": "completed",
            "original_data": data,
            "result": f"processed_{data.get('task', 'no_task')}"
        }

        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)