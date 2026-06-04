from typing import Callable, Any, Dict

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

            data = self.jobs[job_id]
            max_retries = 3

            while self.retry_counts[job_id] <= max_retries:
                try:
                    processor(data)
                    return True
                except Exception as e:
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Calculate backoff: 2^retry_count (1, 2, 4)
                    # Wait/Simulate wait
                    self.retry_counts[job_id] += 1
                    # Logic for backoff delay if needed for tracking:
                    # delay = 2**(self.retry_counts[job_id] - 1)
                    # But requirements say "backoff delays can be stored/tracked rather than actually sleeping"
                    # So I'll just increment the count.
            return False