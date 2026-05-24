from typing import Callable, Any, Dict, Optional
   import time

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           attempts = 0

           while attempts <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   # Success
                   self.retry_counts[job_id] = attempts  # or 0 if successful on first try? The requirement says "track retry count", so maybe track total attempts or retries. I'll track retries (0 if succeeded immediately, 1 if failed once, etc.)
                   # Actually, let's track the number of retries attempted. If it succeeds on first try, retries = 0.
                   self.retry_counts[job_id] = attempts - 1 if attempts > 0 else 0
                   return True
               except Exception as e:
                   attempts += 1
                   if attempts > max_retries:
                       self.retry_counts[job_id] = max_retries
                       return False
                   # Exponential backoff: 1, 2, 4
                   delay = 2 ** (attempts - 1)
                   self.backoff_delays[job_id].append(delay)
                   # Simulate tracking rather than sleeping
                   # time.sleep(delay) # Not needed per requirements
           return False