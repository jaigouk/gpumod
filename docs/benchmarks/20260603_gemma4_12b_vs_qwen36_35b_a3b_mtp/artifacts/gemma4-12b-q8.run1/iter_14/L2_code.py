import time
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
                        # Exponential backoff: 2^attempt (0->1, 1->2, 2->4)
                        delay = 2 ** attempt
                        # time.sleep(delay) # Simulating delay
                        # print(f"Retry {self.retry_counts[job_id]} after {delay}s")
                        pass
                    else:
                        return False
            return False