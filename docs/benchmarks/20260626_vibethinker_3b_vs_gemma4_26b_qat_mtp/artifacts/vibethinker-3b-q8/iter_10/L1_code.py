from collections import deque
from typing import Dict, Optional

class JobQueue:
    def __init__(self) -> None:
        self._queue: deque[Dict[str, object]] = deque()

    def add_job(self, job_id: str, data: Dict) -> str:
        # Assume the caller provides unique job_id; otherwise, a set can be added to enforce uniqueness.
        self._queue.append({"job_id": job_id, "data": data})
        return job_id

    def get_result(self, job_id: str) -> Optional[Dict]:
        while self._queue:
            current = self._queue[0]
            if current["job_id"] == job_id:
                # Result is considered the provided data (actual processing can be inserted here)
                return current["data"]
            else:
                # Consider the front job as processed and move forward
                self._queue.popleft()
        return None