from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            while True:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    current_retries = self.retry_counts[job_id]
                    if current_retries >= max_retries:
                        return False
                    
                    self.retry_counts[job_id] += 1
                    # Backoff calculation: 2^(retry_count - 1)
                    # retry_count 1 -> 1s, 2 -> 2s, 3 -> 4s
                    delay = 2 ** (self.retry_counts[job_id] - 1)
                    # Simulated sleep
                    pass