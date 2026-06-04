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

        def process_next(self) -> Optional[dict]:
            if not self.queue:
                return None
            job_id = self.queue.popleft()
            # Simulate processing or provide a way to set result
            # Since I don't know the logic, I'll just mark it as "done"
            # or let the user provide the result.
            # This is getting complicated.