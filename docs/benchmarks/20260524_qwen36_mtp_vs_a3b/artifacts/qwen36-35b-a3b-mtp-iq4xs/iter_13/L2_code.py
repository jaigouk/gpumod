import time
   from typing import Callable, Dict, Any, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: dict):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0
           current_delay = base_delay

           for attempt in range(max_retries + 1):  # 1 initial + 3 retries
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   self.retry_counts[job_id] = attempt + 1
                   if attempt < max_retries:
                       self.backoff_delays[job_id].append(current_delay)
                       # Simulate backoff (can use time.sleep or just store)
                       # The prompt says "can be simulated", so I'll just store it, but maybe add time.sleep for realism or skip it. I'll skip actual sleep to keep it testable, or use time.sleep but note it's optional. Actually, I'll just store the delays as requested: "The backoff delays can be stored/tracked rather than actually sleeping."
                       current_delay *= 2
                   else:
                       return False