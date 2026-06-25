from collections import deque

class JobQueue:
    def __init__(self):
        self._pending = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._pending.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def complete_job(self, job_id: str, result: dict) -> None:
        self._results[job_id] = result