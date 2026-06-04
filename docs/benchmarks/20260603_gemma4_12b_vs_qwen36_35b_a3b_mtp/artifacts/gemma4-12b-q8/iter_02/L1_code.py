from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.jobs = {}
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.jobs[job_id] = data
            self.queue.append(job_id)
            return job_id

        def process_next_job(self) -> None:
            if self.queue:
                job_id = self.queue.popleft()
                data = self.jobs.pop(job_id)
                # Simulate processing
                self.results[job_id] = {"status": "completed", "data": data}

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)