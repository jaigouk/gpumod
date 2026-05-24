from collections import deque
from typing import Optional, Dict, Any

class JobQueue:
    def __init__(self):
        self.queue = deque()
        self.results = {}
        
    def add_job(self, job_id: str, data: dict) -> str:
        self.queue.append((job_id, data))
        return job_id
        
    def get_result(self, job_id: str) -> Optional[dict]:
        return self.results.get(job_id)
        
    def process_next(self) -> Optional[dict]:
        if not self.queue:
            return None
        job_id, data = self.queue.popleft()
        # Simulate processing
        result = {"job_id": job_id, "status": "completed", "data": data}
        self.results[job_id] = result
        return result