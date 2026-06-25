from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self):
        self._pending_queue = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = {'data': data, 'processed': False}
        self._pending_queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def set_result(self, job_id: str, result: dict) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]['processed'] = True
            self._results[job_id] = result
            if self._pending_queue and self._pending_queue[0] == job_id:
                self._pending_queue.popleft()

    def get_next_job(self) -> Optional[str]:
        if self._pending_queue:
            return self._pending_queue[0]
        return None