from collections import deque

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.queue = deque()

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self.jobs:
            return job_id

        self.jobs[job_id] = {"data": data, "result": None}
        self.queue.append(job_id)
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        if job_id not in self.jobs:
            return None

        job_info = self.jobs[job_id]
        return job_info.get("result")