import time
   from typing import Callable, Dict, Any

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = [1, 2, 4]

       def process_job(self, job_id: str, processor: Callable) -> bool:
           # If job not in queue, maybe raise or handle? Example implies it's added first.
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           while self.retry_counts[job_id] <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if self.retry_counts[job_id] > max_retries:
                       return False
                   # Store backoff delay instead of sleeping
                   delay = self.backoff_delays[job_id].pop(0)
                   # Actually, let's just store the delay for tracking
                   # The requirement says delays can be stored/tracked rather than actually sleeping.
                   # I'll just record it in a list or attribute.
           return False