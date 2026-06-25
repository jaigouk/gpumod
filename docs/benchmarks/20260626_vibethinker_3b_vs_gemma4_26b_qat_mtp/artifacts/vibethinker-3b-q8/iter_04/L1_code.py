from collections import deque
from typing import Dict, Any, Optional

class JobQueue:
    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> str:
        self._queue.append(job_id)
        self._jobs[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self._jobs:
            result = self._jobs.pop(job_id)
            self._queue.remove(job_id)
            return result
        return None