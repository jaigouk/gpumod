import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           backoff_base = 1  # seconds

           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

           for attempt in range(max_retries + 1):  # 1 initial + 3 retries = 4 attempts total? Wait, requirement says "retry up to 3 times". Usually means 1 initial + up to 3 retries = 4 total attempts. Let's stick to 3 retries after initial failure.
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       delay = backoff_base * (2 ** attempt)  # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                   else:
                       return False