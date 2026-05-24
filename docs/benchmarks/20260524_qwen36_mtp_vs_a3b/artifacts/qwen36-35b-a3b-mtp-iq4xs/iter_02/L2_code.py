import time
   from typing import Callable, Dict, Any

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           # Check if job exists
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           # Exponential backoff delays: 1, 2, 4
           backoff_sequence = [1, 2, 4]

           for attempt in range(max_retries + 1):  # 0 to 3 (3 retries after first attempt)
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception as e:
                   if attempt < max_retries:
                       delay = backoff_sequence[attempt]
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate backoff (no actual sleep)
                       # time.sleep(delay)
                   else:
                       return False