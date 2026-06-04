from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._seen_ids = set()

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        Returns the job_id.
        """
        if job_id in self._seen_ids:
            raise ValueError("Job ID must be unique")

        self._queue.append({"id": job_id, "data": data})
        self._seen_ids.add(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job, or None if not completed.
        """
        return self._results.get(job_id)

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Marks a job as completed and stores its result.
         This is used to transition a job from 'pending' to 'completed'.
        """
        self._results[job_id] = result

        # Remove from the internal queue to signify it is processed
        for i, item in enumerate(self._queue):
            if item["id"] == job_id:
                del self._queue[i]
                break