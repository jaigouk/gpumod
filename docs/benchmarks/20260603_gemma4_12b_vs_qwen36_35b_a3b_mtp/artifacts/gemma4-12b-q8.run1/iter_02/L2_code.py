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
                    return True
                except Exception as e:
                    if attempt == max_retries:
                        return False

                    self.retry_counts[job_id] += 1
                    # Delay = 2^(count-1)
                    # attempt 0 fails -> count=1, delay=1
                    # attempt 1 fails -> count=2, delay=2
                    # attempt 2 fails -> count=3, delay=4
                    # attempt 3 fails -> return False
                    delay = 2 ** (self.retry_counts[job_id] - 1)
                    print(f"Retry {self.retry_counts[job_id]} for {job_id} in {delay}s...")
                    # Simulate sleep
            return False