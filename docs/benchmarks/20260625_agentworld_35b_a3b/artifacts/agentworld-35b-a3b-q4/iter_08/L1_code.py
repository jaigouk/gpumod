import collections

class JobQueue:
    def __init__(self):
        self.pending_jobs = collections.deque()
        self.completed_results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append((job_id, data))
        if job_id not in self.completed_results:
            self.completed_results[job_id] = None
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.completed_results.get(job_id)

    def process_next(self) -> str | None:
        if not self.pending_jobs:
            return None
        job_id, data = self.pending_jobs.popleft()
        self.completed_results[job_id] = {"status": "completed", "data": data}
        return job_id