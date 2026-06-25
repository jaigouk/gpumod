from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._pending_jobs = deque()
        self._completed_results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._pending_jobs.append({'job_id': job_id, 'data': data})
        return job_id

    def process_next(self) -> Optional[dict]:
        if not self._pending_jobs:
            return None
        job = self._pending_jobs.popleft()
        job_id = job['job_id']
        data = job['data']
        result = {'status': 'completed', 'data': data}
        self._completed_results[job_id] = result
        return result

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._completed_results.get(job_id)