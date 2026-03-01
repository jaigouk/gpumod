from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = {
            "data": data,
            "retry_count": 0,
            "backoff_delays": []
        }

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        attempts = job["retry_count"]
        max_retries = 3

        while attempts < max_retries:
            try:
                processor(job["data"])
                return True
            except Exception:
                attempts += 1
                if attempts <= max_retries:
                    delay = 2 ** (attempts - 1)
                    job["backoff_delays"].append(delay)
                    job["retry_count"] = attempts
                else:
                    return False
        
        return False