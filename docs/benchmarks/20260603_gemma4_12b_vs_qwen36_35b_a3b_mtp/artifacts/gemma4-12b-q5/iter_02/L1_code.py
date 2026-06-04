from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue: deque[tuple[str, dict]] = deque()
        self._results: Dict[str, dict] = {}
        self._known_ids: set[str] = set()

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._known_ids:
            raise ValueError(f"Job ID '{job_id}' already exists.")

        self._queue.append((job_id, data))
        self._known_ids.add(job_id)
        return job_id

    def process_next_job(self) -> None:
        """
        Processes the next job in the queue (FIFO).
        This method is necessary to satisfy the FIFO requirement 
        and allow get_result to return data.
        """
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing logic
            self._results[job_id] = {"status": "completed", "data": data}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

# Example usage:
if __name__ == "__main__":
    queue = JobQueue()
    job_id = queue.add_job("job1", {"task": "process_data"})

    # Simulate processing
    queue.process_next_job()

    result = queue.get_result("job1")
    print(result)