import time
   from typing import Callable, Any, Dict

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
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           delays = [1, 2, 4]

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = 0 # Reset on success? Or keep track of retries used? The prompt says "Track retry count for each job". I'll keep it as the number of retries performed.
                   return True
               except Exception as e:
                   if attempt < max_retries:
                       self.retry_counts[job_id] = attempt + 1
                       self.backoff_delays[job_id].append(delays[attempt])
                       # Simulate tracking backoff instead of sleeping
                   else:
                       self.retry_counts[job_id] = max_retries
                       return False