import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {} # Or just track delays used

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = [] # Track applied backoff delays

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           backoff_times = [1, 2, 4]

           for attempt in range(max_retries + 1): # 0 initial + 3 retries = 4 attempts total? Wait, requirement says "retry up to 3 times". Usually means 1 initial + 3 retries = 4 attempts, or just 3 attempts total? I'll assume 1 initial + 3 retries = 4 attempts, or just loop 3 times total? Let's stick to standard: attempt 0 (initial), then retry 1, 2, 3. So 4 attempts total. Or maybe just 3 attempts total? The requirement says "retry up to 3 times", which typically means 3 retries after the first failure. I'll implement 4 attempts total (1 initial + 3 retries).
           # Actually, to be safe, I'll loop `range(4)` for attempts, or explicitly handle retries.
           # Let's track retry count separately.
           # I'll use a loop for attempts up to 3 retries.