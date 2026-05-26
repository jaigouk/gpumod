from typing import Callable, Dict, Any, Optional

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_delays: Dict[str, float] = {}
            self.max_retries = 3
            self.backoff_sequence = [1, 2, 4] # seconds

        def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0
            self.backoff_delays[job_id] = 0.0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            if job_id not in self.jobs:
                return False

            job_data = self.jobs[job_id]
            retry_count = self.retry_counts.get(job_id, 0)
            max_attempts = self.max_retries + 1  # Initial + 3 retries? Or just 3 total?
            # Requirement says "retry up to 3 times". Usually means 1 initial + 3 retries.
            # Let's assume 1 initial + 3 retries = 4 calls.
            # Backoff sequence has 3 items.
            # Retry 1 -> index 0 (1s)
            # Retry 2 -> index 1 (2s)
            # Retry 3 -> index 2 (4s)

            attempts = 0
            while attempts < self.max_retries + 1:
                try:
                    processor(job_data)
                    self.retry_counts[job_id] = attempts
                    return True
                except Exception:
                    attempts += 1
                    if attempts <= self.max_retries:
                        # Store backoff delay for next attempt
                        delay_idx = attempts - 1
                        if delay_idx < len(self.backoff_sequence):
                            delay = self.backoff_sequence[delay_idx]
                            self.backoff_delays[job_id] = delay
                            self.retry_counts[job_id] = attempts
                            # Simulate wait
                    else:
                        # Max retries reached
                        break
            return False