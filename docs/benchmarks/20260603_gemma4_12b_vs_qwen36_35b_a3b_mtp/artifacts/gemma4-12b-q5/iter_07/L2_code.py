from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Any] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Any):
            self.jobs[job_id] = data
            self.retry_counts[job_id] = 0

        def process_job(self, job_id: str, processor: Callable) -> bool:
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    processor(self.jobs[job_id])
                    return True
                except Exception:
                    if attempt < max_retries:
                        # Exponential backoff: 2^attempt
                        # Attempt 0 (1st fail) -> backoff 2^0 = 1
                        # Attempt 1 (2nd fail) -> backoff 2^1 = 2
                        # Attempt 2 (3rd fail) -> backoff 2^2 = 4
                        self.retry_counts[job_id] = attempt + 1
                        # Simulate backoff (no actual sleep)
                        pass
                    else:
                        return False
            return False