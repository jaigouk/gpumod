import time
    from typing import Callable, Any, Dict, Optional

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_delays: Dict[str, float] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.backoff_delays[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            base_backoff = 1
            attempts = 0

            while attempts <= max_retries:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    self.retry_counts[job_id] += 1
                    attempts += 1
                    if attempts <= max_retries:
                        # Calculate backoff: 2^(attempts-1) -> 1, 2, 4
                        delay = base_backoff * (2 ** (attempts - 1))
                        self.backoff_delays[job_id] = delay
                    else:
                        self.backoff_delays[job_id] = 0 # No wait after last fail
            return False