from collections import deque
    from typing import Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append((job_id, data))
            return job_id

        def process_next(self):
            if not self.queue:
                return
            job_id, data = self.queue.popleft()
            # Simulate processing: here we just return the data as the result
            self.results[job_id] = {"status": "completed", "processed_data": data}

        def get_result(self, job_id: str) -> Optional[dict]:
            return self.results.get(job_id)