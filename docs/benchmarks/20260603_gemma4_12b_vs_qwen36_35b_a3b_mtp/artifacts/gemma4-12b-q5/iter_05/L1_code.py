from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.results[job_id] = data # Simulating result for now
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)