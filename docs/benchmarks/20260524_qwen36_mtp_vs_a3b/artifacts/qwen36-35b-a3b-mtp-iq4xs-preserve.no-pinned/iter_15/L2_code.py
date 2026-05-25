import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}
           self.max_retries = 3
           self.base_backoff = 1.0

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = self.max_retries
           current_retries = 0

           while current_retries <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   # Success
                   return True
               except Exception:
                   current_retries += 1
                   if current_retries <= max_retries:
                       backoff_time = self.base_backoff * (2 ** (current_retries - 1))
                       self.backoff_delays[job_id].append(backoff_time)
                       # Simulate sleep or just track it
                       # time.sleep(backoff_time) # Commented out as per "can be stored/tracked rather than actually sleeping"
                   else:
                       return False