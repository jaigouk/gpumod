import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0  # seconds
           current_retry = 0

           while current_retry <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if current_retry < max_retries:
                       delay = base_delay * (2 ** current_retry)
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = current_retry + 1
                       # Simulate sleep by just storing delay
                       # time.sleep(delay) # Not used per requirement
                       current_retry += 1
                   else:
                       return False
           return False