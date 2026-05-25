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
           base_delay = 1.0  # seconds

           # Check if job exists
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           for attempt in range(max_retries + 1):  # 1 initial + 3 retries = 4 attempts total? Wait, requirement says "retry up to 3 times". Usually means initial attempt + 3 retries = 4 total attempts, or 3 total attempts. Let's stick to 3 retries after the first attempt. I'll interpret "retry up to 3 times" as up to 3 additional attempts after the first. So total attempts = 4. But sometimes it means 3 total. I'll do 3 retries (total 4 attempts) to be safe, or just 3 attempts total. Actually, standard is: attempt 1, if fails, retry 1, retry 2, retry 3. So 4 attempts total. I'll stick with `range(max_retries + 1)` or just `range(3)` for retries. Let's re-read: "retry up to 3 times". I'll do 3 retries.
           # Wait, exponential backoff: 1s, 2s, 4s. That's 3 delays. So 3 retries.
           # I'll track retries and delays.

           # Let's implement carefully:
           for i in range(4):  # 1 initial + 3 retries
               try:
                   processor(self.jobs[job_id])
                   # Success
                   self.retry_counts[job_id] = i  # or track actual retries
                   return True
               except Exception:
                   if i < 3:  # If not the last retry
                       delay = 2 ** i  # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = i + 1
                   else:
                       self.retry_counts[job_id] = 3
                       return False
           return False