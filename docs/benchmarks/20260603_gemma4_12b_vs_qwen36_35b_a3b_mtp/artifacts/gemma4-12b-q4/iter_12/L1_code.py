from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        # Store jobs as a deque to ensure FIFO order
        self._queue: deque[tuple[str, dict]]] = deque()
        # Store results indexed by job_id
        self._results: Dict[str, Any] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the unique job_id."""
        self._queue.append((job_id, data))
        self._results[job_id] = None
        return job_id

    def process_next(self) -> None:
        """Processes the next job in the FIFO queue."""
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate job processing logic
            self._results[job_id] = {"status": "completed", "processed_data": data}

    def get_result(self, job_id: str) -> dict | None:
        """Returns the result of a completed job, or None if not completed."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate processing
    queue.process_next()

    result = queue.get_result("job1")
    print(f"Job ID: {job_id}, Result: {result}")