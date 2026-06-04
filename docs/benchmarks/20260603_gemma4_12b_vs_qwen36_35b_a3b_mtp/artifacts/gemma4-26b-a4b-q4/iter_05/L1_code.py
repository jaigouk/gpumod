from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue and returns the job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> Optional[dict]:
        """
        Processes the next job in the queue (FIFO).
        Simulates processing by returning a result dictionary.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        # Simulating processing logic
        result = {"status": "completed", "original_data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[dict]:
        """Gets the result of a completed job."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the worker processing the job
    queue.process_next_job()

    # Get the result
    result = queue.get_result("job1")
    print(result)