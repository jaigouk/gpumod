import time
    from typing import Callable, Any, Dict

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.delay_history: Dict[str, list] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.delay_history[job_id] = []

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            max_retries = 3
            data = self.jobs[job_id]

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception:
                    if self.retry_counts[job_id] < max_retries:
                        # Calculate exponential backoff: 1, 2, 4
                        delay = 2 ** self.retry_counts[job_id]
                        self.delay_history[job_id].append(delay)
                        self.retry_counts[job_id] += 1
                        # Simulation of sleep: we don't actually sleep as per instructions
                        # but we track the delay.
                    else:
                        self.retry_counts[job_id] += 1 # To mark as exhausted
                        break
            return False