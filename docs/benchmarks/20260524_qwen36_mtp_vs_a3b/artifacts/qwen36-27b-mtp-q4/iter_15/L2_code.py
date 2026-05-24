from typing import Callable, Dict, Any, Optional
   import time # Not strictly needed if simulating, but might be good for structure. I'll skip actual sleep.

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {} # To track delays if needed, or just use a fixed list

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False # Or raise KeyError, but returning False is safer per spec

           job_data = self.jobs[job_id]
           max_retries = 3
           delays = [1, 2, 4]

           for attempt in range(max_retries):
               try:
                   processor(job_data)
                   self.retry_counts[job_id] = 0 # Reset on success? Or just leave it. The prompt says "Track retry count for each job", probably means during processing. I'll reset or just track current attempts. Let's track total attempts or current retry count. I'll reset to 0 on success, or just keep it as is. Actually, it's better to just track the number of retries used.
                   return True
               except Exception:
                   if attempt < len(delays):
                       self.backoff_delays[job_id].append(delays[attempt])
                   self.retry_counts[job_id] = attempt + 1
                   # No actual sleep, just track/simulate
           return False