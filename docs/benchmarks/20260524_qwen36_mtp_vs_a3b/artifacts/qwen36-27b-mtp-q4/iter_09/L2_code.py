import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}  # Track delays per job

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           current_retry = self.retry_counts[job_id]

           while current_retry <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   # Success: reset retry count or keep it? The requirement says "Track retry count for each job"
                   # I'll reset it on success, or maybe just leave it. Let's reset to 0 on success.
                   self.retry_counts[job_id] = 0
                   self.backoff_delays[job_id] = []
                   return True
               except Exception:
                   if current_retry < max_retries:
                       # Calculate backoff: 1s, 2s, 4s
                       delay = 2 ** current_retry  # 2^0=1, 2^1=2, 2^2=4
                       self.backoff_delays[job_id].append(delay)
                       # Simulate tracking rather than sleeping
                       # self.retry_counts[job_id] += 1
                       current_retry += 1
                   else:
                       return False
           return False