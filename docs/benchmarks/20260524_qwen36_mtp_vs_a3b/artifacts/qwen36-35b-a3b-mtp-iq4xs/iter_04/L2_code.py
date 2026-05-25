import time
   from typing import Callable, Dict, Any

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
           backoff_sequence = [1, 2, 4]
           job_data = self.jobs[job_id]

           while self.retry_counts[job_id] <= max_retries:
               try:
                   processor(job_data)
                   self.retry_counts[job_id] = 0  # Reset on success
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if self.retry_counts[job_id] <= max_retries:
                       delay = backoff_sequence[self.retry_counts[job_id] - 1]
                       self.backoff_delays[job_id].append(delay)
                   else:
                       return False
           return False