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

            data = self.jobs[job_id]
            max_retries = 3

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    # Success
                    self.retry_counts[job_id] = 0 # Reset or keep? Usually reset on success.
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff logic: 1, 2, 4
                        # attempt 0 fail -> retry_counts becomes 1 -> backoff 2^(1-1) = 1
                        # attempt 1 fail -> retry_counts becomes 2 -> backoff 2^(2-1) = 2
                        # attempt 2 fail -> retry_counts becomes 3 -> backoff 2^(3-1) = 4
                        backoff = 2 ** (self.retry_counts[job_id] - 1)
                        # print(f"Retry {self.retry_counts[job_id]} after {backoff}s")
                    else:
                        return False
            return False