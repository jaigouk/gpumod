import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           backoff_sequence = [1.0, 2.0, 4.0]  # 1s, 2s, 4s

           for attempt in range(max_retries + 1):  # +1 for initial try
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt < max_retries:
                       self.retry_counts[job_id] = attempt + 1
                       self.backoff_delays[job_id].append(backoff_sequence[attempt])
                   else:
                       return False