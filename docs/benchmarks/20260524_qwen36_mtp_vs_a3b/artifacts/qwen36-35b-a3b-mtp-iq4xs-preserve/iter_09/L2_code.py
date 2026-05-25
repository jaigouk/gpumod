import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = {"data": data}

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_backoff = 1.0
           retries = 0
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

           while retries <= max_retries:
               try:
                   processor(self.jobs[job_id]["data"])
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if retries < max_retries:
                       delay = base_backoff * (2 ** retries)
                       self.backoff_delays[job_id].append(delay)
                       # Simulate backoff instead of sleeping
                       # time.sleep(delay)
                   retries += 1

           return False