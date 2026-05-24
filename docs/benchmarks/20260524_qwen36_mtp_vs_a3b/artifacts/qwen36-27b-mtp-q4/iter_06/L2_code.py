from typing import Callable, Dict, Any, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {} # To track simulated delays

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
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
                   # Calculate backoff: 1s, 2s, 4s
                   delay = 2 ** (self.retry_counts[job_id] - 1)
                   self.backoff_delays[job_id].append(delay)
           return False