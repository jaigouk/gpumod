import threading
from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    def add_job(self, job_id: str, data: dict) -> str:
        with self._lock:
            self._queue.append(job_id)
            self._results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        with self._lock:
            return self._results.get(job_id)

    def process(self) -> dict | None:
        with self._lock:
            if not self._queue:
                return None
            job_id = self._queue.popleft()
            self._results[job_id] = {"status": "completed", "data": self._results[job_id]}
            return self._results[job_id]