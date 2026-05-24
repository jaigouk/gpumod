from collections import deque

class JobQueue:
    def __init__(self):
        self.queue = deque()
        self.results = {}
        
    def add_job(self, job_id: str, data: dict) -> str:
        self.queue.append({"job_id": job_id, "data": data})
        return job_id
        
    def process(self):
        if not self.queue:
            return None
        job = self.queue.popleft()
        # Simulate processing
        result = {"job_id": job["job_id"], "data": job["data"], "status": "completed"}
        self.results[job["job_id"]] = result
        return result
        
    def get_result(self, job_id: str) -> dict | None:
        return self.results.get(job_id, None)