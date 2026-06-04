from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Processes the next job in the queue using FIFO order.
        In a real scenario, this might run in a background thread/process.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {
            "status": "success",
            "processed_data": data
        }

        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the result of a completed job."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the processing step
    queue.process_next_job()

    result = queue.get_result("job1")
    print(result)