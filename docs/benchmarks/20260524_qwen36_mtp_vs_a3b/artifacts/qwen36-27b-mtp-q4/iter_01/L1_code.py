from collections import deque
import uuid
import time

class JobQueue:
    def __init__(self):
        self._queue = deque()
        self._results = {}
        
    def add_job(self, job_id: str, data: dict) -> str:
        self._queue.append((job_id, data))
        return job_id
        
    def get_result(self, job_id: str) -> dict | None:
        return self._results.get(job_id)
        
    def process_next(self):
        if not self._queue:
            return None
        job_id, data = self._queue.popleft()
        # Simulate processing
        result = {"status": "completed", "data": data}
        self._results[job_id] = result
        return result