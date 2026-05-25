import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       delay = base_delay * (2 ** attempt)
                       self.backoff_delays[job_id].append(delay)
                       # Simulate sleep by tracking delay instead of time.sleep()
                       # Or just use time.sleep if allowed, but prompt says can be stored/tracked
                       # I'll just record it. To be safe, I'll add a small pass or just not sleep.
                       # Actually, I'll just record it. The prompt says "can be stored/tracked rather than actually sleeping"
                   else:
                       return False
           return False