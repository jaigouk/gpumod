from typing import Callable, Dict, Any

    class JobQueue:
        def __init__(self):
            self.jobs: Dict[str, Dict[str, Any]] = {}
            self.retry_counts: Dict[str, int] = {}

        def add_job(self, job_id: str, data: Dict[str, Any]):
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
                    self.retry_counts[job_id] += 1
                    if self.retry_counts[job_id] > max_retries:
                        return False
                    
                    # Simulate exponential backoff (1, 2, 4)
                    # Attempt 0 fails -> retry 1 (wait 1s)
                    # Attempt 1 fails -> retry 2 (wait 2s)
                    # Attempt 2 fails -> retry 3 (wait 4s)
                    # Attempt 3 fails -> fail
                    # The loop range(4) covers attempts 0, 1, 2, 3.
                    # If attempt 0, 1, 2 fail, it retries.
                    # Actually, "retry up to 3 times" usually means 1 initial try + 3 retries = 4 tries total.
                    # The prompt says "retry up to 3 times".
                    # Backoff delays: 1s, 2s, 4s.
                    # If attempt 0 fails -> backoff 2**0 = 1
                    # If attempt 1 fails -> backoff 2**1 = 2
                    # If attempt 2 fails -> backoff 2**2 = 4
                    # If attempt 3 fails -> loop ends -> False.
                    pass