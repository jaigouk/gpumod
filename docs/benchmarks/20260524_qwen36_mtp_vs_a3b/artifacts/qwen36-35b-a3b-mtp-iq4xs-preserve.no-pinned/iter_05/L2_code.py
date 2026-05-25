import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = {
               "data": data,
               "retries": 0,
               "backoff_delays": []
           }

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           job = self.jobs[job_id]
           max_retries = 3
           base_backoff = 1  # seconds

           for attempt in range(max_retries + 1):
               try:
                   processor(job["data"])
                   # Success
                   return True
               except Exception:
                   job["retries"] += 1
                   if job["retries"] > max_retries:
                       return False
                   # Calculate backoff for next attempt
                   backoff = base_backoff * (2 ** (job["retries"] - 1))
                   job["backoff_delays"].append(backoff)
                   # Simulate sleep by storing, or just continue
                   # The prompt says "can be simulated", so I'll just store it.
                   # Actually, I can just store the delay or skip time.sleep()
                   # To be safe, I'll just continue the loop.
           return False