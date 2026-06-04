from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            self.results[job_id] = None
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)

        def complete_job(self, job_id: str, result: dict):
            self.results[job_id] = result