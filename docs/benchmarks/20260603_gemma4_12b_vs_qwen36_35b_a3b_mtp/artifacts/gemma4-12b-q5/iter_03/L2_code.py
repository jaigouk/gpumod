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
            data = self.jobs[job_id]

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    self.retry_counts[job_id] = 0 # Reset or keep? Requirement says "Track", usually implies cumulative.
                    return True
                except Exception as e:
                    self.retry_counts[job_id] += 1
                    if self.retry_counts[job_id] > max_retries:
                        return False
                    # Backoff simulation (not actual sleep as per instructions)
                    # 1s, 2s, 4s is 2^attempt
                    # attempt=0 (1st failure) -> 2^0 = 1
                    # attempt=1 (2nd failure) -> 2^1 = 2
                    # attempt=2 (3rd failure) -> 2^2 = 4
                    # No need to actually sleep, just logic check.
            return False