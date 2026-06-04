from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}
            self.jobs = {} # To keep track of input data

        def add_job(self, job_id: str, data: dict) -> str:
            if job_id in self.jobs or job_id in self.results:
                raise ValueError("Job ID must be unique")
            self.jobs[job_id] = data
            self.queue.append(job_id)
            return job_id

        def process_next_job(self):
            if self.queue:
                job_id = self.queue.popleft()
                # Simulate processing
                data = self.jobs.pop(job_id)
                self.results[job_id] = {"status": "completed", "data": data}

        def get_result(self, job_id: str) -> Optional[dict]]:
            return self.results.get(job_id)