import time
from typing import Callable, Any, Dict

class JobQueue:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.retry_counts: Dict[str, int] = {}
        self.backoff_delays: Dict[str, list] = {}
        
    def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
        self.jobs[job_id] = data
        self.retry_counts[job_id] = 0
        self.backoff_delays[job_id] = []
        
    def process_job(self, job_id: str, processor: Callable) -> bool:
        if job_id not in self.jobs:
            return False
            
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries + 1):  # 1 initial + 3 retries = 4 attempts? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 total, or maybe just 3 attempts total. Let's assume 3 retries means 3 additional attempts after the first failure. Actually, typical interpretation: up to 3 retries means total 4 attempts, but sometimes it means 3 attempts total. I'll stick with 3 retries (4 attempts total) or just 3 attempts total. The prompt says "retry up to 3 times", so initial + 3 retries.
        # Wait, exponential backoff: 1s, 2s, 4s. That's 3 delays, which matches 3 retries.
        # So: attempt 1 (no delay), fail -> delay 1s, attempt 2, fail -> delay 2s, attempt 3, fail -> delay 4s, attempt 4, fail -> return False.
        # Let's implement exactly that.
        
        for attempt in range(max_retries + 1):
            try:
                processor(self.jobs[job_id])
                self.retry_counts[job_id] = attempt
                return True
            except Exception:
                self.retry_counts[job_id] = attempt
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.backoff_delays[job_id].append(delay)
                    # Simulate sleep or just track it
                    # time.sleep(delay) # The prompt says "can be simulated", so I'll just track it
                else:
                    return False