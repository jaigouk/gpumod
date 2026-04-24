from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"job_id": job_id, "data": data})
        return job_id

    def process_next(self):
        if self._queue:
            job = self._queue.popleft()
            job_id = job["job_id"]
            self._results[job_id] = {"status": "completed", "data": job["data"]}

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)