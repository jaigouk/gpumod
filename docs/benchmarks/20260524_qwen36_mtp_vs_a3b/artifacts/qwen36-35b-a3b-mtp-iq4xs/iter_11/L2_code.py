import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retries: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retries[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           for attempt in range(max_retries):
               try:
                   processor(self.jobs[job_id])
                   self.retries[job_id] = attempt
                   return True
               except Exception as e:
                   if attempt < max_retries - 1:
                       delay = 2 ** attempt  # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       # Simulate sleep without actually waiting
                       # time.sleep(delay) 
                       continue
                   else:
                       self.retries[job_id] = attempt
                       return False
           return False