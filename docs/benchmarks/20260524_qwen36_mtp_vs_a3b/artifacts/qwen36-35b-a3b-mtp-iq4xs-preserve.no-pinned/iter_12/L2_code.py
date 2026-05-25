import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = {
               "data": data,
               "retry_count": 0,
               "backoff_delays": []
           }

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           job = self.jobs[job_id]
           max_retries = 3
           backoff_delays = [1, 2, 4]  # seconds

           for attempt in range(max_retries + 1):  # 1 initial + 3 retries
               try:
                   processor(job["data"])
                   job["retry_count"] = attempt
                   return True
               except Exception as e:
                   job["retry_count"] = attempt
                   if attempt < max_retries:
                       delay = backoff_delays[attempt]
                       job["backoff_delays"].append(delay)
                       # Simulate backoff by storing instead of sleeping
                   else:
                       return False