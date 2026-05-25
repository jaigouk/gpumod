import collections

class JobQueue:
    def __init__(self):
        self.queue = collections.deque()
        self.results = {}
        self.job_data = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.queue.append(job_id)
        self.job_data[job_id] = data
        return job_id

    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id)

    def process(self):
        if self.queue:
            job_id = self.queue.popleft()
            # Simulate processing
            self.results[job_id] = {"status": "completed", "data": self.job_data[job_id]}