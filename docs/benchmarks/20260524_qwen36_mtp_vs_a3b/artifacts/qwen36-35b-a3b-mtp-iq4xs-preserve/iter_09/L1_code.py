import collections
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._results: Dict[str, dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)

    def process_next(self) -> Optional[dict]:
        if not self._queue:
            return None
        job_id, data = self._queue.popleft()
        # Simulate processing
        result = {"status": "completed", "processed_data": data}
        self._results[job_id] = result
        return result