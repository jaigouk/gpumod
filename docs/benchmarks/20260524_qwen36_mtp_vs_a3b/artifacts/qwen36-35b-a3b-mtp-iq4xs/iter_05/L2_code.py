import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: List[float] = [1.0, 2.0, 4.0]
           self.retry_delays: Dict[str, List[float]] = {} # To track used delays per job

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.retry_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception as e:
                   self.retry_counts[job_id] = attempt + 1
                   if attempt < max_retries:
                       delay = self.backoff_delays[attempt]
                       self.retry_delays[job_id].append(delay)
                       # Simulate backoff by storing, not sleeping
                       # time.sleep(delay) # explicitly not sleeping per requirements
                   else:
                       return False