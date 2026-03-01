from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: Dict[str, Optional[Dict]] = {}
        self._job_data: Dict[str, Dict] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._results:
            raise ValueError(f"Job ID '{job_id}' already exists")
        self._queue.append(job_id)
        self._job_data[job_id] = data
        self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        return self._results.get(job_id)

    def process_next_job(self) -> Optional[str]:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        data = self._job_data[job_id]
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return job_id