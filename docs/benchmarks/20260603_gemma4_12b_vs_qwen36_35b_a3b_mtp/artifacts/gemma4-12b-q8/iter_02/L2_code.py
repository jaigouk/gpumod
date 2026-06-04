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
                    processor(self.jobs[job_id]))
                    self.retry_counts[job_id] = 0 # Reset on success
                    return True
                except Exception:
                    self.retry_counts[job_id] += 1
                    if self.retry_counts[job_id] > max_retries:
                        return False
                    
                    # Simulate backoff calculation
                    # retry 1 -> 2^0 = 1s
                    # retry 2 -> 2^1 = 2s
                    # retry 3 -> 2^2 = 4s
                    # Wait... The requirement says 1s, 2s, 4s.
                    # If attempt 0 fails, it's retry 1.
                    # Let's look at the requirement: "retry up to 3 times".
                    # If it fails first time, retry 1.
                    # If it fails second time, retry 2.
                    # If it fails third time, retry 3.
                    # Total 4 attempts possible.
                    
                    # Calculation:
                    # attempt 0 (fail) -> retry 1 -> backoff 2^(1-1) = 1
                    # attempt 1 (fail) -> retry 2 -> backoff 2^(2-1) = 2
                    # attempt 2 (fail) -> retry 3 -> backoff 2^(3-1) = 4
                    # attempt 3 (fail) -> return False
                    
                    # So backoff = 2 ** (self.retry_counts[job_id] - 1)
                    # But I don't need to actually sleep.
                    pass
            return False