from collections import deque
from typing import Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: dict[str, dict[str, Any]] = {}

    def add_job(self, job_id: str, data: dict[str, Any]) -> str:
        """Add a job to the queue, return job_id."""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self) -> dict[str, Any] | None:
        """
        Processes the next job in the queue (FIFO).
        In a real-world scenario, this might be handled by a separate worker thread.
        """
        if not self._queue:
            return None

        job_id, data = self._queue.popleft()

        # Simulate processing logic
        result = {
            "job_id": job_id,
            "status": "completed",
            "processed_data": data
        }

        self._results[job_id] = result
        return result

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Get the result of a completed job."""
        return self._results.get(job_id)

# Example usage
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the background worker processing the job
    queue.process_next_job()

    # Get the result
    result = queue.get_result("job1")
    print(result)