from typing import Callable, Dict, Any

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}

    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0

    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False

        retry_count = 0
        max_retries = 3
        backoff_delays = [1, 2, 4]  # seconds

        while retry_count <= max_retries:
            try:
                processor(self.jobs[job_id])
                # Success
                self.retry_counts[job_id] = 0
                return True
            except Exception:
                if retry_count < max_retries:
                    delay = backoff_delays[retry_count]
                    # Track backoff delay instead of sleeping
                    self.jobs[job_id]['last_backoff'] = delay
                    self.retry_counts[job_id] = retry_count + 1
                    retry_count += 1
                else:
                    self.retry_counts[job_id] = retry_count + 1
                    return False
        
        return False