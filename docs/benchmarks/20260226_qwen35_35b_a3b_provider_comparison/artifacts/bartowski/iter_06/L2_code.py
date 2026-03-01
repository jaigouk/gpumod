import time
from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Any] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, list] = {}
        self.max_retries = 3
        self.backoff_sequence = [1, 2, 4]

    def add_job(self, job_id: str, data: Dict[str, Any]):
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        retry_count = self.retry_counts[job_id]
        data = self.jobs[job_id]

        while retry_count <= self.max_retries:
            try:
                processor(data)
                return True
            except Exception:
                self.retry_counts[job_id] = retry_count + 1
                if retry_count < len(self.backoff_sequence):
                    delay = self.backoff_sequence[retry_count]
                    self.backoff_delays[job_id].append(delay)
                    # time.sleep(delay) # Simulated
                    retry_count += 1
                else:
                    return False
        return False