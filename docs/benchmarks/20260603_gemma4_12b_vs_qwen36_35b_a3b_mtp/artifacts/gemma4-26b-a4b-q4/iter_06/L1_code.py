from collections import deque

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            return job_id

        def process_next_job(self, worker_func):
            """Helper to actually process a job and move it to results."""
            if not self.queue:
                return
            job_id, data = self.queue.popleft()
            result = worker_func(data)
            self.results[job_id] = result

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)