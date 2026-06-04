from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Adds a job to the FIFO queue."""
        self._queue.append((job_id, data))
        return job_id

    def process_next(self) -> None:
        """
        Processes the next job in the queue.
        In a real-world scenario, this would involve executing task logic.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing by returning the data back with a status
            self._results[job_id] = {
                "status": "completed",
                "original_data": data
            }

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the result of a completed job, or None if not found/not completed."""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()

    # Add jobs
    id1 = queue.add_job("job1", {"task": "process_data"})
    id2 = queue.add_job("job2", {"task": "send_email"})

    # Process jobs in FIFO order
    queue.process_next()  # Processes job1
    queue.process_next()  # Processes job2

    # Get results
    print(f"Result 1: {queue.get_result(id1)}")
    print(f"Result 2: {queue.get_result(id2)}")
    print(f"Result 3 (non-existent): {queue.get_result('job3')}")