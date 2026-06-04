from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            self.results[job_id] = None # Initialize as not done
            return job_id

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)

        def process_next_job(self):
            if self.queue:
                job_id, data = self.queue.popleft()
                # Simulate processing
                self.results[job_id] = {"status": "completed", "result": data}