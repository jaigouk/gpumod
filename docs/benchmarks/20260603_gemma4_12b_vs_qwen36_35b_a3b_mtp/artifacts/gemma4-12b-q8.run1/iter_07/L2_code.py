from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs = {} # job_id -> data
            self.retry_counts = {} # job_id -> count

        def add_job(self, job_id: str, data: Dict[str, Any]):
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
                except Exception:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Simulate backoff
                        delay = 2 ** (self.retry_counts[job_id] - 1)
                        # print(f"Retry {self.retry_counts[job_id]} for {job_id} after {delay}s")
                    else:
                        break

            return False