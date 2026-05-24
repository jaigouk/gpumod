import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           backoffs = [1, 2, 4]
           data = self.jobs.get(job_id)
           if data is None:
               return False

           for attempt in range(max_retries):
               try:
                   processor(data)
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if attempt < max_retries - 1:
                       # Simulate backoff by storing it or just passing
                       # The prompt says "backoff delays can be stored/tracked rather than actually sleeping"
                       # I'll store the delay in a dict or just use time.sleep if needed, but prompt says can be simulated.
                       # I'll just use a placeholder or store it. Actually, I'll just implement the logic without sleep, or use a configurable sleep that defaults to simulated.
                       # To be safe, I'll just not sleep, or store the delay. The prompt says "can be simulated", so I'll just note it or use a dummy sleep.
                       # Actually, I'll just implement the retry loop. I'll store the backoff delay in a dict for tracking.
                       self._backoff_delays[job_id] = backoffs[attempt]
                       # time.sleep(backoffs[attempt]) # Optional, but prompt says can be simulated
           return False