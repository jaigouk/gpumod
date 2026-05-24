import time
   from typing import Callable, Dict, Any, Optional
   from collections import defaultdict

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = defaultdict(int)
           self.backoff_delays: Dict[str, list] = defaultdict(list)

       def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
           self.jobs[job_id] = data

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           backoff_sequence = [1, 2, 4]

           for attempt in range(max_retries):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries - 1:
                       delay = backoff_sequence[attempt]
                       self.backoff_delays[job_id].append(delay)
                       # Simulate backoff by just tracking, not sleeping
                       # If actual sleep is needed, time.sleep(delay)
                   else:
                       # Last attempt failed
                       pass

           return False