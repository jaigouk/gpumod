from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # FIFO queue to store job IDs
        self._queue: deque[str] = deque()
        # Dictionary to store job data while pending
        self._jobs: Dict[str, dict] = {}
        # Dictionary to store results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        """
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def complete_job(self, job_id: str, result: dict):
        """
        Helper method to mark a job as completed and store its result.
         Required to transition job state for get_result to work.
        """
        self._results[job_id] = result
        if job_id in self._jobs:
            del self._jobs[job_id]
        if job_id in self._queue:
            self._queue.remove(job_id)

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Returns the result of a completed job, or None if not completed.
        """
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate processing logic
    queue.complete_job("job1", {"status": "success", "output": "processed_data"})

    result = queue.get_result("job1")
    print(result)