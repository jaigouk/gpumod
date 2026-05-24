from typing import Callable, Any, Dict

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {} # To track backoff delays

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           delays = [1, 2, 4] # Exponential backoff delays

           for attempt in range(max_retries + 1): # 1 initial + 3 retries = 4 attempts total? Wait, requirement says "retry up to 3 times". Usually means 1 initial attempt + 3 retries = 4 total attempts. Or does it mean 3 total attempts? Let's assume 3 retries after initial failure, so 4 attempts total. But sometimes "retry up to 3 times" means 3 attempts total. I'll stick to 3 retries (4 attempts) or clarify by implementing exactly 3 retries. Actually, standard is: attempt 1, if fails retry 1, retry 2, retry 3. Total 4 attempts. I'll implement 3 retries.
           # Wait, let's re-read: "retry up to 3 times". I'll do 3 retries.
           # Actually, to be safe, I'll make it exactly 3 retries.
           # Let's track attempts: initial + retries.
           # I'll use a loop for retries.

           # Better approach:
           for attempt in range(4): # 1 initial + 3 retries
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < 3: # Can still retry
                       delay = 2 ** attempt # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                   else:
                       return False