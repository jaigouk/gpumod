from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """
        Adds a job to the queue in FIFO order.
        """
        self._queue.append({"id": job_id, "data": data})
        return job_id

    def process_next(self) -> None:
        """
        Processes the next job in the queue and stores the result.
        """
        if self._queue:
            job = self._queue.popleft()
            job_id = job["id"]
            data = job["data"]

            # Simulate processing logic
            result = {
                "status": "completed",
                "processed_data": data
            }
            self._results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        """
        Returns the result of a completed job by its ID.
        """
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate the processing step
    queue.process_next()

    result = queue.get_result("job1")
    print(result)