from typing import Callable, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, int] = {}

    def add_job(self, job_id: str, data: dict):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False
        
        self.retry_counts[job_id] = 0
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                return True
            except Exception:
                self.retry_counts[job_id] += 1
                if attempt < max_retries:
                    delay = 2 ** attempt
                    self.backoff_delays[job_id] = delay
                else:
                    return False
        return False