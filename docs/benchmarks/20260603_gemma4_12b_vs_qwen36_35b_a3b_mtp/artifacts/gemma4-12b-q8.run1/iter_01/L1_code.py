import collections
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = collections.deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append({"id": job_id, "data": data})
            return job_id

        def complete_job(self, job_id: str, result: dict):
            # Added to make the example functional
            self.results[job_id] = result

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)