import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: list = [1, 2, 4]

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       self.retry_counts[job_id] += 1
                       # Simulate backoff delay
                       delay = self.backoff_delays[attempt]
                       # Store the delay instead of sleeping, as requested
                       # But we need to track it. The prompt says "The backoff delays can be stored/tracked rather than actually sleeping."
                       # I'll just record the delay in a dict or just proceed without sleep.
                       # Actually, to simulate, I could just not sleep, or store it in a list.
                       # Let's store the applied backoff per job.
                       if not hasattr(self, 'backoff_log'):
                           self.backoff_log = {}
                       self.backoff_log[job_id] = delay
                   else:
                       return False
           return False