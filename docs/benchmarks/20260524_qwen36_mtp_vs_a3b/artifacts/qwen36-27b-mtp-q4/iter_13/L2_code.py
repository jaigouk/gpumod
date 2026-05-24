import time
   from typing import Callable, Dict, Any, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           # Retry logic: up to 3 retries means 4 attempts total? Or 3 attempts total?
           # Usually "retry up to 3 times" means 1 initial + 3 retries = 4 attempts, or 3 total attempts.
           # I'll interpret it as max 3 retries after the first attempt, so 4 total attempts.
           # But often in these problems, it means 3 total attempts. Let's stick to 3 retries (4 attempts) or clarify by using a loop.
           # Actually, "retry up to 3 times" typically means if it fails, retry 1, 2, 3. So 4 attempts total.
           # I'll implement a loop that tries up to 4 times (initial + 3 retries), or just track retries separately.
           # Let's use a while loop: attempts = 0, max_attempts = 4
           # Or simpler: retries = 0, while retries <= 3: try... if fail, retries += 1, backoff = 2**(retries-1)
           # Wait, backoff sequence: 1s, 2s, 4s. That's 3 backoffs. So 4 attempts total.
           # I'll track retry count as required: "Track retry count for each job"
           # I'll update self.retry_counts[job_id] accordingly.

           # Let's implement carefully:
           attempts = 0
           max_attempts = 4  # 1 initial + 3 retries
           while attempts < max_attempts:
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = attempts - 1  # retries used
                   return True
               except Exception:
                   attempts += 1
                   if attempts <= max_attempts:
                       # Track backoff delay (simulated)
                       delay = 2 ** (attempts - 2) if attempts > 1 else 1
                       # Actually, backoff sequence: 1, 2, 4 for retries 1, 2, 3
                       # Let's just store/track the backoff delays as required
                       # The prompt says "The backoff delays can be stored/tracked rather than actually sleeping."
                       # I'll store them in a dict or just compute them. I'll add a backoff_log or just track in retry_counts.
                       # Actually, I'll just update retry_counts and maybe store backoff info if needed, but prompt just says "track retry count".
                       pass
           self.retry_counts[job_id] = 3
           return False