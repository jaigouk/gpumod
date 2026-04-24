from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}
        self._results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self._jobs[job_id] = data
        self._queue.append(job_id)
        return job_id

    def process(self) -> str | None:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        data = self._jobs.pop(job_id)
        # Simulate job processing
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)