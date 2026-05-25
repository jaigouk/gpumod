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
           base_delay = 1.0

           # Initial attempt
           try:
               processor(job["data"])
               return True
           except Exception:
               pass

           # Retry logic
           for retry_num in range(max_retries):
               delay = base_delay * (2 ** retry_num)  # 1, 2, 4
               job["backoff_delays"].append(delay)
               job["retry_count"] += 1
               # Simulate waiting (optional, but prompt says can be stored/tracked)
               # time.sleep(delay)
               try:
                   processor(job["data"])
                   return True
               except Exception:
                   pass

           return False