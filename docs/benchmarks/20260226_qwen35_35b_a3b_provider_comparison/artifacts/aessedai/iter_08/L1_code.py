from collections import deque

class JobQueue:
    def __init__(self):
        self.queue = deque()
        self.results = {}
        self.seen_ids = set()

    def add_job(self, job_id: str, data: dict) -> str:
        if job_id in self.seen_ids:
            raise ValueError("Job ID already exists")
        self.queue.append({'job_id': job_id, 'data': data, 'status': 'pending'})
        self.seen_ids.add(job_id)
        return job_id

    def process(self):
        while self.queue:
            job = self.queue.popleft()
            job['status'] = 'completed'
            self.results[job['job_id']] = job

    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id)