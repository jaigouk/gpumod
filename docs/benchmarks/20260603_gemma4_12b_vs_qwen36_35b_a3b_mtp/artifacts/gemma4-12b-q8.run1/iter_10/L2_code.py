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
                    self.retry_counts[job_id] = 0 # Reset on success
                    return True
                except Exception as e:
                    if attempt < max_retries:
                        self.retry_counts[job_id] += 1
                        # Backoff: 1s (attempt 0), 2s (attempt 1), 4s (attempt 2)
                        # Wait... if it fails on attempt 0, we retry.
                        # If it fails on attempt 1, we retry.
                        # If it fails on attempt 2, we retry.
                        # If it fails on attempt 3, we stop.
                        
                        # Wait, the prompt says "retry up to 3 times".
                        # That means 4 total attempts? Or 3 total?
                        # "Retry up to 3 times" usually means 1 initial + 3 retries = 4 attempts.
                        # Backoff values: 1s, 2s, 4s.
                        # If it fails first time (attempt 0), backoff 1s.
                        # If it fails second time (attempt 1), backoff 2s.
                        # If it fails third time (attempt 2), backoff 4s.
                        # If it fails fourth time (attempt 3), return False.
                        
                        # Let's calculate delay: 2**attempt
                        # Attempt 0 fail -> delay 1
                        # Attempt 1 fail -> delay 2
                        # Attempt 2 fail -> delay 4
                        # Attempt 3 fail -> return False
                        pass # Simulated sleep
                    else:
                        return False
            return False