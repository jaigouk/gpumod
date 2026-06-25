from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append({"job_id": job_id, "data": data})
        return job_id

    def process(self) -> dict | None:
        if self._queue:
            job = self._queue.popleft()
            result = {"status": "completed", "data": job["data"]}
            self._results[job["job_id"]] = result
            return result
        return None

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)