import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, float] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = 0.0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           base_delay = 1.0
           current_delay = 0.0

           for attempt in range(max_retries + 1):  # 0, 1, 2, 3 -> 4 attempts total? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 total attempts, or initial + 3 retries. Let's interpret as: initial attempt + up to 3 retries. So max_retries = 3, total attempts = 4.
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = attempt
                   return True
               except Exception:
                   if attempt < max_retries:
                       current_delay = base_delay ** attempt  # Wait, 1s, 2s, 4s. That's 2^0, 2^1, 2^2? Or 1, 2, 4. Yes, 2^(attempt) for attempt 1,2,3? Let's check: first retry after 1s, second after 2s, third after 4s.
                       # Actually, standard exponential backoff: delay = base * (2 ** attempt) or base * (attempt ** something).
                       # Requirement says: "1s, 2s, 4s". So delays are [1, 2, 4].
                       # If attempt=0 (first fail), delay=1. attempt=1 (second fail), delay=2. attempt=2 (third fail), delay=4.
                       # So delay = 2 ** attempt
                       self.backoff_delays[job_id] = 2 ** attempt
                       self.retry_counts[job_id] = attempt + 1
                   else:
                       self.retry_counts[job_id] = max_retries
                       return False
           return False