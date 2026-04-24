from collections import deque

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._jobs: dict[str, dict] = {}
        
    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append(job_id)
        self._jobs[job_id] = {"data": data, "result": None}
        return job_id
        
    def get_result(self, job_id: str) -> dict | None:
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job["result"] is None:
                job["result"] = job["data"]
            return job["result"]
        return None