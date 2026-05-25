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
           backoff_delays = [1.0, 2.0, 4.0]
           retry_count = 0

           while retry_count <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if retry_count < max_retries:
                       self.backoff_delays[job_id].append(backoff_delays[retry_count])
                       retry_count += 1
                       # Simulate backoff by storing instead of sleeping
                       # Or just track it as required
                   else:
                       return False