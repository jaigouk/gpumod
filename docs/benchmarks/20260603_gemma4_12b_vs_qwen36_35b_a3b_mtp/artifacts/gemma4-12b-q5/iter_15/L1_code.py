from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.jobs = {} # Stores data
            self.results = {} # Stores results

        def add_job(self, job_id: str, data: dict) -> str:
            self.jobs[job_id] = data
            self.queue.append(job_id)
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)

        def process_next(self) -> Optional[dict]:
            if not self.queue:
                return None
            job_id = self.queue.popleft()
            data = self.jobs.get(job_id)
            # Simulate processing
            result = {"status": "completed", "original_data": data}
            self.results[job_id] = result
            return result