from collections import deque
from typing import Dict, Optional, Tuple

class JobQueue:
    def __init__(self):
        self._job_queue = deque()
        self._jobs: Dict[str, dict] = {}
        self._results: Dict[str, Optional[dict]] = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id not in self._jobs:
            self._jobs[job_id] = {'data': data, 'processed': False}
            self._job_queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id)

    def set_result(self, job_id: str, result: dict) -> None:
        if job_id in self._jobs:
            self._jobs[job_id]['processed'] = True
            self._results[job_id] = result
            self._cleanup_queue()

    def get_next_job(self) -> Optional[Tuple[str, dict]]:
        self._cleanup_queue()
        if self._job_queue:
            job_id = self._job_queue[0]
            return job_id, self._jobs[job_id]['data']
        return None

    def _cleanup_queue(self) -> None:
        while self._job_queue:
            job_id = self._job_queue[0]
            if not self._jobs.get(job_id, {}).get('processed', False):
                break
            self._job_queue.popleft()