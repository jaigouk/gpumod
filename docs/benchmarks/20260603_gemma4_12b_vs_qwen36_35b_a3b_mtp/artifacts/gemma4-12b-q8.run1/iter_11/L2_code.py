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
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff calculation: 2^(count-1)
                        # attempt 0 (first fail) -> count 1 -> delay 2^0 = 1
                        # attempt 1 ( second fail) -> count 2 -> delay 2^1 = 2
                        # attempt 2 ( third fail) -> count 3 -> delay 2^2 = 4
                        delay = 2 ** (self.retry_counts[job_id] - 1)
                        print(f"Retry {self.retry_counts[job_id]} for {job_id} after {delay}s delay")
                        # time.sleep(delay) # Simulated
                    else:
                        return False
            return False