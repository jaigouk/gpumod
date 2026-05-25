from collections import deque
import uuid
from typing import Dict, Optional, Any

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        
    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id
        
    def process(self):
        if self._queue:
            job_id, data = self._queue.popleft()
            # Simulate processing
            self._results[job_id] = {"status": "completed", "data": data}
            
    def get_result(self, job_id: str) -> Optional[dict]:
        return self._results.get(job_id, None)