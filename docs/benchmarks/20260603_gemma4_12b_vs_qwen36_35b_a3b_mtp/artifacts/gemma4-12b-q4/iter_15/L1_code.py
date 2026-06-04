from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        # Stores the order of jobs to be processed (FIFO)
        self._queue: deque[str] = deque()
        # Stores the raw job data
        self._jobs: Dict[str, dict] = {}
        # Stores the final results of processed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        """
        if job_id in self._jobs or job_id in self._results:
            raise ValueError(f"Job ID {job_id} already exists.")

        self._jobs[job_id] = data
        self._queue.append(job_id)
        self._results[job_id] = None
        return job_id

    def complete_job(self, job_id: str, result: dict):
        """
        Helper method to simulate the processing of a job 
         and store its result.
        """
        if job_id in self._jobs:
            self._results[job_id] = result
            # Remove from internal job storage once processed
            del self._jobs[job_id]
            # Remove from queue if it's still there
            if job_id in self._queue:
                self._queue.remove(job_id)

    def get_result(self, job_id: str) -> dict | None:
        """
        Retrieves the result of a completed job.
        """
        return self._results.get(job_id)