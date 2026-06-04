from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}
            self.backoff_history: Dict[str, list] = {} # Optional, but helps tracking

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

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
                    if self.retry_counts[job_id] == max_retries:
                        return False

                    # Calculate backoff: 1, 2, 4
                    delay = 2 ** self.retry_counts[job_id]
                    # Requirement: "The backoff delays can be stored/tracked rather than actually sleeping"
                    # We can just simulate or just calculate.
                    self.retry_counts[job_id] += 1
                    # In a real app, time.sleep(delay)
                    # To satisfy "can be simulated", I'll just increment the count and continue.

            return False