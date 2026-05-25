import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, dict] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: dict):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0  # seconds

           # Ensure job exists
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           current_retries = self.retry_counts.get(job_id, 0)

           for attempt in range(max_retries):
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = 0  # Reset on success? Or keep as is. Requirement says "Track retry count for each job". I'll just track it.
                   return True
               except Exception as e:
                   # Calculate backoff delay
                   delay = base_delay * (2 ** attempt)  # 1, 2, 4 for attempt 0, 1, 2
                   self.backoff_delays[job_id].append(delay)
                   self.retry_counts[job_id] = attempt + 1
                   # Simulate backoff by storing instead of sleeping
                   # If actual sleep is needed, we could add it, but requirement says "can be simulated"
                   # I'll stick to storing it as requested.

           return False