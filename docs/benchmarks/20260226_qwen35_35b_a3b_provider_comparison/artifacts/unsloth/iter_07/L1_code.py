from collections import deque

class JobQueue:
    def __init__(self):
        self._jobs = deque()
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs.append((job_id, data))
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)

    def process(self):
        while self._jobs:
            job_id, data = self._jobs.popleft()
            self._results[job_id] = {"status": "completed", "input_data": data}