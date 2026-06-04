from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.job_data = {}
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.job_data[job_id] = data
            return job_id

        def process_next(self) -> str | None:
            if not self.queue:
                return None
            job_id = self.queue.popleft()
            data = self.job_data.pop(job_id)
            # Simulate processing
            self.results[job_id] = {"status": "completed", "data": data}
            return job_id

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)