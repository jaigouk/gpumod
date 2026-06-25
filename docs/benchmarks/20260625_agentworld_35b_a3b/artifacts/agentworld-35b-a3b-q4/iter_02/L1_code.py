import collections

class JobQueue:
    def __init__(self):
        self.job_queue = collections.deque()
        self.job_data = {}
        self.job_results = {}

    def add_job(self, job_id: str, data: dict) -> str:
        self.job_queue.append(job_id)
        self.job_data[job_id] = data
        if job_id not in self.job_results:
            self.job_results[job_id] = None
        return job_id

    def get_next_job(self) -> tuple[str, dict] | None:
        if self.job_queue:
            job_id = self.job_queue.popleft()
            return job_id, self.job_data.get(job_id)
        return None

    def complete_job(self, job_id: str, result: dict):
        self.job_results[job_id] = result

    def get_result(self, job_id: str) -> dict | None:
        return self.job_results.get(job_id)