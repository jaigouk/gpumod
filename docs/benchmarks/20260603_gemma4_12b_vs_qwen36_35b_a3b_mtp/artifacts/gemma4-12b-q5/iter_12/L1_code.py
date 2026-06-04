from collections import deque
    from typing import Dict, Optional

    class JobQueue:
        def __init__(self):
            self.queue = deque()
            self.results = {}

        def add_job(self, job_id: str, data: dict) -> str:
            self.queue.append(job_id)
            self.results[job_id] = {"data": data, "result": None, "completed": False}
            return job_id

        def process_next_job(self):
            if self.queue:
                job_id = self.queue.popleft()
                # Simulate processing
                self.results[job_id]["result"] = f"Processed: {self.results[job_id]['data']}"
                self.results[job_id]["completed"] = True

        def get_result(self, job_id: str) -> dict | None:
            job = self.results.get(job_id)
            if job and job["completed"]:
                return job["result"]
            return None