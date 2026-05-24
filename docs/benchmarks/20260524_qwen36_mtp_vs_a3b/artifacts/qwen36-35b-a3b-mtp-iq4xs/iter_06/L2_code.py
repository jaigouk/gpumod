import time
   from typing import Callable, Any, Dict, List

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
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           data = self.jobs[job_id]
           max_retries = 3
           backoff_sequence = [1, 2, 4]

           for attempt in range(max_retries + 1):  # 0 to 3 (4 attempts total? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 attempts, or just 3 attempts total. Let's assume 3 retries means max 4 attempts. But the backoff is 1, 2, 4. That's 3 backoffs. So: attempt 0 -> if fail, wait 1, retry 1 -> if fail, wait 2, retry 2 -> if fail, wait 4, retry 3 -> if fail, return False. That matches 3 retries.)
               try:
                   processor(data)
                   # Success
                   return True
               except Exception:
                   if attempt < max_retries:
                       delay = backoff_sequence[attempt]
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate waiting
                       # time.sleep(delay) # Requirement says can be simulated/stored
                   else:
                       self.retry_counts[job_id] = max_retries
                       return False
           return False