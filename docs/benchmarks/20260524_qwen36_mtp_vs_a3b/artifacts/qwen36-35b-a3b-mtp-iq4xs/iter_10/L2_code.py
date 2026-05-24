import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           backoff_values = [1, 2, 4]
           attempts = 0

           while attempts <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   # Success
                   return True
               except Exception as e:
                   attempts += 1
                   if attempts <= max_retries:
                       # Record retry and backoff delay
                       self.retry_counts[job_id] = attempts
                       delay = backoff_values[attempts - 1]
                       self.backoff_delays[job_id].append(delay)
                       # Simulate delay instead of time.sleep
                       # (Could just store it as required)
                   else:
                       # All retries exhausted
                       return False