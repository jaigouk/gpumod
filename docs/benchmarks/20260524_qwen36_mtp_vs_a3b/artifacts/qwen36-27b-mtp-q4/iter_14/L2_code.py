from typing import Callable, Any, Dict, List
   import time # Not needed if simulating

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

           max_retries = 3
           backoff_sequence = [1, 2, 4]

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if attempt < max_retries:
                       delay = backoff_sequence[attempt]
                       self.backoff_delays[job_id].append(delay)
                       # Simulate backoff tracking
                   else:
                       return False