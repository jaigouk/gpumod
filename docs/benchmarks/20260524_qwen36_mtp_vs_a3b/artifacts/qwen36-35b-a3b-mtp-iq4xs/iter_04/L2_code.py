import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = {"data": data, "retry_count": 0}

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           job = self.jobs[job_id]
           max_retries = 3
           succeeded = False

           # Initial attempt
           try:
               processor(job["data"])
               succeeded = True
           except Exception:
               succeeded = False

           # Retry logic
           while not succeeded and job["retry_count"] < max_retries:
               job["retry_count"] += 1
               # Calculate backoff: 1s, 2s, 4s for retries 1, 2, 3
               backoff_delay = 2 ** (job["retry_count"] - 1)
               # Store backoff delay instead of sleeping
               job["next_backoff"] = backoff_delay
               # Simulate delay or just continue
               try:
                   processor(job["data"])
                   succeeded = True
               except Exception:
                   succeeded = False

           return succeeded