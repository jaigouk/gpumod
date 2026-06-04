from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # FIFO queue to store jobs as (job_id, data)
        self._queue: deque = deque()
        # Dictionary to store results keyed by job_id
        self._results: Dict[str, dict | None] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return its unique ID."""
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def process_next_job(self) -> None:
        """
        Helper method to process the next job in the FIFO queue.
         While not explicitly listed in the requirements, it is 
         necessary to fulfill the "jobs should be processed" 
         requirement so that get_result returns a value.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "result": data}

    def get_result(self, job_id: str) -> dict | None:
        """Get the result of a completed job by its ID."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the processing step
    queue.process_next_job()

    result = queue.get_result("job1")
    print(result)