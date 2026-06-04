from collections import deque
from typing import Any, Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Processes the next job in the queue (FIFO).
        Simulates processing by returning the original data with a success status.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()
        # Simulate processing logic
        result = {"status": "completed", "processed_data": data}
        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the result of a completed job."""
        return self._results.get(job_id)

# Example usage matching the prompt's flow:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the "After processing..." step
    queue.process_next_job()

    result = queue.get_result("job1")
    print(result)