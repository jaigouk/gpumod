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
           base_delay = 1.0
           current_delay = base_delay

           # Check if job exists
           if job_id not in self.jobs:
               raise KeyError(f"Job {job_id} not found")

           for attempt in range(max_retries):
               try:
                   processor(self.jobs[job_id])
                   # Success
                   return True
               except Exception:
                   # Track retry
                   self.retry_counts[job_id] = attempt + 1
                   # Track backoff delay (can be simulated)
                   self.backoff_delays[job_id].append(current_delay)
                   # Simulate backoff instead of sleeping
                   # time.sleep(current_delay)
                   # Prepare for next retry
                   current_delay *= 2

           # All retries exhausted
           return False