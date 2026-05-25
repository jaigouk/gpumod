import time
   from typing import Callable, Dict, Any, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = {"data": data}
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1  # seconds

           for attempt in range(max_retries + 1):  # 0 to 3 (4 attempts total)
               try:
                   job_data = self.jobs[job_id]["data"]
                   processor(job_data)
                   # Success
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if attempt < max_retries:
                       delay = base_delay * (2 ** attempt)  # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       # Simulate sleep by just storing it, as per requirement
                   else:
                       return False
           return False