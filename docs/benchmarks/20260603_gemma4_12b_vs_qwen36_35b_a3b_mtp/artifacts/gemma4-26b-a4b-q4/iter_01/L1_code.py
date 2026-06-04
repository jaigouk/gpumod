from collections import deque
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, Any] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        """Add a job to the queue, return job_id"""
        self._queue.append((job_id, data))
        return job_id

    def process_next_job(self, result: dict) -> None:
        """
        Simulates the processing of the next job in the queue.
        Removes the job from the FIFO queue and stores the result.
        """
        if self._queue:
            job_id, _ = self._queue.popleft()
            self._results[job_id] = result

    def get_result(self, job_id: str) -> Optional[dict]:
        """Get the result of a completed job"""
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate a worker processing the job
    queue.process_next_job({"status": "success", "output": [1, 2, 3]})

    # Get the result
    result = queue.get_result("job1")
    print(result)  # {'status': 'success', 'output': [1, 2, 3]}