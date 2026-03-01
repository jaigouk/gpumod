from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job with id '{job_id}' already exists.")
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def process(self):
        """Process all jobs in the queue in FIFO order."""
        while self._queue:
            job_id = self._queue.popleft()
            if job_id in self._jobs:
                data = self._jobs[job_id]
                # Simulate job processing
                self._results[job_id] = {"status": "completed", "original_data": data}
                del self._jobs[job_id]

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

# Example usage (for demonstration purposes)
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})
    queue.process()
    result = queue.get_result("job1")
    print(result)