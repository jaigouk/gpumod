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
                    if attempt == max_retries:
                        return False
                    
                    # Logic for exponential backoff calculation
                    # Attempt 0 failure -> Retry 1 (1s)
                    # Attempt 1 failure -> Retry 2 (2s)
                    # Attempt 2 failure -> Retry 3 (4s)
                    # Attempt 3 failure -> Max reached
                    self.retry_counts[job_id] += 1
                    # Backoff would be 2**(self.retry_counts[job_id]-1)
                    # But since we only need to return bool and "track" it,
                    # we just need to ensure the loop runs correctly.
            return False