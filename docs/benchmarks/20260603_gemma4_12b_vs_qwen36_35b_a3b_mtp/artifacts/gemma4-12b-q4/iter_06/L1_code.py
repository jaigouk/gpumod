from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        # Internal queue to store jobs in FIFO order (deque for O(1) pops from left)
        self._queue: deque[tuple[str, dict]]] = deque()
        # Dictionary to store results of completed jobs
        self._results: Dict[str, dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Add a job to the queue.
        :param job_id: A unique identifier for the job.
        :param data: The data/payload associated with the job.
        :return: The job_id.
        """
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        This method transitions a job from the queue to the results dictionary.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {
                "status": "completed",
                "processed_data": data
            }

    def get_result(self, job_id: str) -> dict | None:
        """
        Get the result of a completed job.
        :param job_id: The unique identifier of the job.
        :return: The result dictionary or None if not completed.
        """
        return self._results.get(job_id)