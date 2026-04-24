from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs = {}

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self._jobs:
            raise ValueError(f"Job with id '{job_id}' already exists")
        self._jobs[job_id] = {"status": "pending", "data": data}
        self._queue.append(job_id)
        return job_id

    def process(self) -> str | None:
        if not self._queue:
            return None
        job_id = self._queue.popleft()
        job = self._jobs[job_id]
        job["status"] = "completed"
        job["result"] = job["data"]
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job and job["status"] == "completed":
            return job["result"]
        return None