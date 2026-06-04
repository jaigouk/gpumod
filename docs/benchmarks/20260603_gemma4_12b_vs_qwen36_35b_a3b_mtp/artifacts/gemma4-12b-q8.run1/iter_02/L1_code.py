from collections import deque

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.jobs = {}
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.jobs[job_id] = data
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)

        def complete_job(self, job_id: str, result: dict):
            self.results[job_id] = result
            # Requirement says "store jobs internally until they are processed"
            # So we remove from the 'jobs' storage.
            if job_id in self.jobs:
                del self.jobs[job_id]