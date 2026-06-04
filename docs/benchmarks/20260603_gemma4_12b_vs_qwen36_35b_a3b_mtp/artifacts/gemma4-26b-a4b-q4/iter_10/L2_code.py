from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_delays: Dict[str, int] = {} # To store the delay used

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    if self.retry_counts[job_id] >= max_retries:
                        return False

                    # Calculate backoff: 2^0=1, 2^1=2, 2^2=4
                    delay = 2 ** self.retry_counts[job_id]
                    self.backoff_delays[job_id] = delay
                    self.retry_counts[job_id] += 1
                    # Simulating sleep as per instructions (not actually sleeping)

            return False