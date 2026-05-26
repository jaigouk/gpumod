import collections
from typing import Optional, Dict, Any

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results: Dict[str, Any] = {}
        self._job_data: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._job_data:
            raise ValueError(f"Job {job_id} already exists")
        self._queue.append(job_id)
        self._job_data[job_id] = data
        return job_id

    def process(self):
        # Simulate processing jobs in FIFO order
        while self._queue:
            job_id = self._queue.popleft()
            # Simulate work by storing the data as result
            # In a real scenario, this would be an external worker
            self._results[job_id] = self._job_data.pop(job_id)

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)