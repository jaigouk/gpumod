from collections import deque

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            return job_id

        def process_next(self, result: dict):
            # Requirement 2: FIFO
            if self.queue:
                job_id, data = self.queue.popleft()
                self.results[job_id] = result

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)