import time
   from typing import Callable, Dict, Any

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Dict[str, Any]):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise KeyError(f"Job {job_id} not found")

           max_retries = 3
           current_retries = self.retry_counts.get(job_id, 0)

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = 0  # Reset on success
                   return True
               except Exception as e:
                   if attempt < max_retries:
                       delay = 2 ** attempt  # 1, 2, 4
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] += 1
                       # Simulate sleep by just tracking, or actually sleep if needed.
                       # Prompt says "can be stored/tracked rather than actually sleeping"
                       # I'll just record it. If they want actual sleep, I'd add time.sleep, but I'll stick to tracking.
                       # Actually, to be safe, I'll just not sleep, or add a comment. I'll just track it.
                   else:
                       return False
           return False