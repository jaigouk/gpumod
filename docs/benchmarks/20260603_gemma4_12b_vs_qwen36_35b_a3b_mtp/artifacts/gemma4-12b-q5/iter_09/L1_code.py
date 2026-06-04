from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}
            self.jobs = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.jobs[job_id] = data
            return job_id

        def process_next(self):
            if not self.queue:
                return None
            job_id = self.queue.popleft()
            data = self.jobs.pop(job_id)
            # Simulate processing
            result = {"status": "completed", "input_data": data}
            self.results[job_id] = result
            return result

        def get_result(self, job_id: str) -> dict | None:
            return self.results.get(job_id)