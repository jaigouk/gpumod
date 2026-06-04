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
            backoffs = [1, 2, 4]

            for attempt in range(max_retries + 1):
                try:
                    processor(data)
                    # Update status if needed, but prompt doesn't specify cleanup.
                    # Just return True on success.
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Simulate backoff (logging or just skipping sleep as per instructions)
                        # print(f"Retry {self.retry_counts[job_id]} after {backoffs[attempt-1]}s")
                        pass
                    else:
                         return False
            return False