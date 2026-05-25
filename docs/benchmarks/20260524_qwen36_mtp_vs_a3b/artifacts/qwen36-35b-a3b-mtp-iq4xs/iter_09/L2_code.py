import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self._jobs: Dict[str, Dict[str, Any]] = {}

       def add_job(self, job_id: str, data: dict) -> None:
           self._jobs[job_id] = {
               "data": data,
               "retry_count": 0,
               "backoff_delays": []
           }

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self._jobs:
               raise ValueError(f"Job {job_id} not found")

           job = self._jobs[job_id]
           max_retries = 3
           delays = [1, 2, 4]

           for attempt in range(max_retries + 1):
               try:
                   processor(job["data"])
                   return True
               except Exception:
                   job["retry_count"] = attempt
                   if attempt < max_retries:
                       job["backoff_delays"].append(delays[attempt])
                   else:
                       job["backoff_delays"].append(delays[attempt]) # or stop here
           return False