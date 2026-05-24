import time
   from typing import Callable, Any, Dict

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
           current_retries = self.retry_counts.get(job_id, 0)

           # We need to handle retries up to 3 times total.
           # The requirement says "retry up to 3 times", which usually means 1 initial attempt + 3 retries = 4 attempts total,
           # or it could mean 3 attempts total. I'll interpret it as 3 retries after the first failure, so max 4 attempts.
           # But to be safe, I'll track attempts and allow up to 3 retries.
           # Actually, "retry up to 3 times" typically means if it fails, try again up to 3 more times.
           # Let's implement a loop that tries, catches exception, increments retry count, applies backoff, and repeats up to 3 retries.

           # Wait, the requirement says "Track retry count for each job". This implies persistent tracking across calls?
           # Or just within the process_job call? The example shows process_job being called once.
           # I'll track it internally and reset or keep it? Usually, retry count is per job processing attempt.
           # I'll keep it in self.retry_counts as requested.

           # Let's implement the retry logic carefully:
           attempts = 0
           max_attempts = 4 # 1 initial + 3 retries
           # Or maybe max_retries = 3 means total attempts = 3? I'll stick to 3 retries after first failure.
           # Actually, it's simpler: loop while attempts <= max_retries (3).
           # Let's use a standard approach:
           retries = 0
           max_retries = 3

           while retries <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = retries # Update to 0 on success? Or keep failed count?
                   # Usually, on success, we might reset or just leave it. I'll set it to 0 on success.
                   self.retry_counts[job_id] = 0
                   return True
               except Exception:
                   retries += 1
                   self.retry_counts[job_id] = retries
                   if retries <= max_retries:
                       # Exponential backoff: 1s, 2s, 4s
                       backoff = 2 ** (retries - 1)
                       # "can be simulated" - I'll just store/log it or use time.sleep(0) to simulate,
                       # but the requirement says "The backoff delays can be stored/tracked rather than actually sleeping."
                       # So I'll just track it, maybe in a separate dict or just compute it.
                       # I'll add a backoff_tracker dict to store the delays.
                       pass
           return False