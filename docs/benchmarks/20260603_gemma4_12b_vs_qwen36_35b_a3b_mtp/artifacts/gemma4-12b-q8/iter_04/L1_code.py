from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # Stores job_ids in FIFO order
        self._queue: deque[str] = deque()
        # Stores the input data for each job_id
        self._jobs: Dict[str, dict] = {}
        # Stores the results of completed jobs
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue.
        Returns the job_id.
        """
        if job_id in self._jobs:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def complete_job(self, job_id: str, result: dict) -> None:
        """
        Helper method to simulate processing a job and storing its result.
        """
        if job_id in self._jobs:
            self._results[job_id] = result

    def get_result(self, job_id: str) -> Optional[dict]:
        """
        Gets the result of a completed job.
        Returns None if the job is not completed.
        """
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate processing...
    queue.complete_job("job1", {"status": "success", "output": 42})

    result = queue.get_result("job1")
    print(result)