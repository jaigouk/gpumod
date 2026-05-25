import time
   from typing import Callable, Any

   class JobQueue:
       def __init__(self):
           self.jobs = {}

       def add_job(self, job_id: str, data: dict) -> None:
           self.jobs[job_id] = {
               "data": data,
               "retry_count": 0,
               "backoff_delay": 0
           }

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           job = self.jobs[job_id]
           max_retries = 3
           backoff_delays = [1, 2, 4]  # seconds

           for attempt in range(max_retries + 1):
               try:
                   result = processor(job["data"])
                   # Success
                   return True
               except Exception as e:
                   job["retry_count"] = attempt + 1
                   if attempt < max_retries:
                       job["backoff_delay"] = backoff_delays[attempt]
                       # Simulate backoff by storing instead of sleeping
                       # time.sleep(job["backoff_delay"])
                   else:
                       # All retries exhausted
                       return False
           return False