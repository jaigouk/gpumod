import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs = {}
           self.retry_counts = {} # Or store inside jobs

       def add_job(self, job_id: str, data: dict):
           self.jobs[job_id] = {
               "data": data,
               "status": "pending",
               "retries": 0,
               "backoff_delays": [] # To track simulated delays
           }

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           job = self.jobs[job_id]
           max_retries = 3
           
           # We need to loop. 
           # Attempt 0 (initial), then retry 1, 2, 3.
           
           for attempt in range(max_retries + 1):
               try:
                   result = processor(job["data"])
                   job["status"] = "completed"
                   return True
               except Exception as e:
                   job["retries"] += 1
                   if attempt < max_retries:
                       # Calculate backoff: 2^(attempt) where attempt is 0, 1, 2
                       # Wait, logic check:
                       # Fail 1 -> attempt 0 -> backoff 1s
                       # Fail 2 -> attempt 1 -> backoff 2s
                       # Fail 3 -> attempt 2 -> backoff 4s
                       # If we are here, we failed. We want to retry.
                       # Delay is 2^attempt
                       delay = 2 ** attempt
                       job["backoff_delays"].append(delay)
                   else:
                       # Max retries reached
                       job["status"] = "failed"
                       return False