from collections import deque

class JobQueue:
    def __init__(self):
        self._pending: deque = deque()
        self._results: dict = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._pending.append({"job_id": job_id, "data": data})
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id, None)

    def process_next(self) -> dict | None:
        if not self._pending:
            return None
        job = self._pending.popleft()
        result = {"status": "completed", "data": job["data"]}
        self._results[job["job_id"]] = result
        return result