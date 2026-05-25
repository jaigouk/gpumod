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
           # Check if job exists
           if job_id not in self.jobs:
               return False

           max_retries = 3
           base_delay = 1.0

           for attempt in range(max_retries + 1):  # 0 to 3 (4 attempts total? Wait, requirement says "retry up to 3 times")
               # Retry up to 3 times means: initial attempt + 3 retries = 4 total attempts? Or does it mean 3 attempts total?
               # Usually "retry up to 3 times" means initial + 3 retries = 4 attempts. But sometimes it means 3 attempts total.
               # Let's stick to initial + 3 retries = 4 attempts, but the backoff is specified as 1s, 2s, 4s. That's exactly 3 backoffs, which matches 3 retries.
               # So attempts: 0 (initial), 1 (retry 1, backoff 1s), 2 (retry 2, backoff 2s), 3 (retry 3, backoff 4s).
               # Wait, backoff happens *between* retries. So:
               # Attempt 0: run
               # If fails, wait 1s, Attempt 1: run
               # If fails, wait 2s, Attempt 2: run
               # If fails, wait 4s, Attempt 3: run
               # If fails, return False.
               # This matches "retry up to 3 times" and backoffs 1, 2, 4.

               # Actually, let's track retries properly:
               # retry_count starts at 0.
               # We try up to 3 retries.
               # Let's implement a loop that runs at most 4 times (initial + 3 retries).
               pass

           # Let's refine the loop structure:
           attempts = 0
           max_attempts = 4  # 1 initial + 3 retries
           current_delay = 0.0

           # Wait, the requirement says "retry up to 3 times" and "exponential backoff between retries: 1s, 2s, 4s".
           # So:
           # Attempt 1: run
           # Fail -> record delay 1s, retry
           # Attempt 2: run
           # Fail -> record delay 2s, retry
           # Attempt 3: run
           # Fail -> record delay 4s, retry
           # Attempt 4: run
           # Fail -> exhausted
           # Success -> return True

           # I'll track retry count and delays as requested.
           # The requirement says "Track retry count for each job". I'll update self.retry_counts[job_id].
           # "The backoff delays can be stored/tracked rather than actually sleeping."

           # Let's write it out carefully.