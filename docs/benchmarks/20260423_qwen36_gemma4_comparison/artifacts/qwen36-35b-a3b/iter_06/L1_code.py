from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        self._pending = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._pending or job_id in self._results:
            raise ValueError("Duplicate job_id")
        self._pending[job_id] = data
        self._queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)