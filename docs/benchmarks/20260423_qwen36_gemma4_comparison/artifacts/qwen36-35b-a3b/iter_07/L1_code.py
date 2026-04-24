from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id

    def process(self) -> str | None:
        if self._queue:
            job_id, data = self._queue.popleft()
            self._results[job_id] = {"status": "completed", "data": data}
            return job_id
        return None

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)