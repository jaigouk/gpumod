import time
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
            for attempt in range(max_retries + 1):
                try:
                    data = self.jobs[job_id]
                    processor(data)
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff logic
                        # 1s, 2s, 4s corresponds to 2^(attempt)
                        delay = 2 ** attempt
                        # Simulated sleep as per requirements
                        print(f"Retry {self.retry_counts[job_id]} for {job_id} after {delay}s")
                    else:
                        return False
            return False