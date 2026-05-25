import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           backoff_sequence = [1, 2, 4]

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       self.retry_counts[job_id] += 1
                       delay = backoff_sequence[attempt]
                       self.backoff_delays[job_id].append(delay)
                       # Simulate backoff instead of time.sleep
                   else:
                       return False