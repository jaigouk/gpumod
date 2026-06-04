from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        # Internal queue to store (job_id, data) tuples
        self._queue: deque[tuple[str, dict]]] = deque()
        # Dictionary to store completed results
        self._results: Dict[str, dict] = {}
        # Set to ensure job_id uniqueness
        self._existing_ids: set[str] = set()

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue. 
        Ensures the job_id is unique.
        """
        if job_id in self._existing_ids:
            raise ValueError(f"Job ID '{job_id}' already exists in the queue.")

        self._existing_ids.add(job_id)
        self._queue.append((job_id, data))
        return job_id

    def process_jobs(self) -> None:
        """
        Processes all jobs currently in the queue in FIFO order.
        This method facilitates the 'processing' step required by the prompt.
        """
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic: 
            # In a real scenario, this would call a worker function.
            self._results[job_id] = {
                "status": "completed",
                "processed_data": data
            }

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job, or None if not yet processed.
        """
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the processing step
    queue.process_jobs()

    result = queue.get_result("job1")
    print(result)