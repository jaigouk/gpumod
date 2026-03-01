from typing import Callable

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.retry_counts = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        job_data = self.jobs[job_id]
        max_retries = 3
        backoff_delays = [1, 2, 4]
        
        for attempt in range(max_retries + 1):
            try:
                processor(job_data)
                self.retry_counts[job_id] = attempt
                return True
            except Exception:
                if attempt < max_retries:
                    # Simulate exponential backoff (1s, 2s, 4s)
                    # Tracking delay instead of sleeping as per requirements
                    delay = backoff_delays[attempt]
                else:
                    self.retry_counts[job_id] = max_retries
                    return False
        
        return False