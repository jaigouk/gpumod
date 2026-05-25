from typing import Callable, Any, Dict, List, Optional
   import time

   class JobQueue:
       def __init__(self):
           self.jobs = {}  # job_id -> data
           self.retry_counts = {}  # job_id -> int
           self.backoff_delays = [1, 2, 4]  # seconds

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           for attempt in range(max_retries + 1):  # 0 to 3 (1 initial + 3 retries)
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate backoff delay (store it, don't sleep)
                       delay = self.backoff_delays[attempt]
                       # Could store in a dict or just use it
                   else:
                       return False
           return False