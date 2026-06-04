from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        # FIFO queue to store jobs as (job_id, data) tuples
        self._queue: deque[tuple[str, dict]] = deque()
        # Dictionary to store results keyed by job_id
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue and return the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_jobs(self):
        """
        Processes all jobs in the queue in FIFO order.
        This method is provided to satisfy the requirement that 
        jobs are processed and moved from the internal queue to results.
        """
        while self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic (e.g., transformation or computation)
            self._results[job_id] = {
                "status": "completed",
                "input_data": data,
                "output": f"Processed: {data.get('task', 'no_task')}"
            }

    def get_result(self, job_id: str) -> dict | None:
        """Retrieve the result of a completed job by its ID."""
        return self._results.get(job_id)

# Example Usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the processing step
    queue.process_jobs()

    result = queue.get_result("job1")
    print(result)