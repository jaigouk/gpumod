import collections

class JobQueue:
    def __init__(self):
        self.pending_jobs = collections.deque()
        self.job_data = {}
        self.completed_results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.pending_jobs.append(job_id)
        self.job_data[job_id] = data
        return job_id

    def get_next_job(self) -> str | None:
        if self.pending_jobs:
            return self.pending_jobs.popleft()
        return None

    def complete_job(self, job_id: str, result: dict) -> None:
        if job_id in self.job_data:
            self.completed_results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self.completed_results.get(job_id)