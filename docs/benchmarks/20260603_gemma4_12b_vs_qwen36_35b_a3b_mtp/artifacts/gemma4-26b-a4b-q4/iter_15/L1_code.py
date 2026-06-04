from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.jobs = {}  # job_id -> data
            self.results = {} # job_id -> result

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.jobs[job_id] = data
            return job_id

        def process_next_job(self):
            if not self.queue:
                return
            job_id = self.queue.popleft()
            data = self.jobs.pop(job_id)
            # Simulate processing: just returning the data as the result
            result = {"status": "completed", "processed_data": data}
            self.results[job_id] = result

        def get_result(self, job_id: str) -> Optional[Dict]:
            return self.results.get(job_id)