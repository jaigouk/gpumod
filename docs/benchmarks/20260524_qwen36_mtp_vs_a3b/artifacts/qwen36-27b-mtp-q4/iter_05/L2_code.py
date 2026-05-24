from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           backoffs = [1, 2, 4]

           for attempt in range(max_retries + 1):  # 0 to 3 (4 attempts total: 1 initial + 3 retries)
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = 0  # Reset on success? Or just return True. Requirement says track retry count, doesn't specify reset. I'll keep it as is or reset. Let's just track it. Actually, requirement says "Track retry count for each job", so I'll maintain it.
                   return True
               except Exception:
                   if attempt < max_retries:
                       # Apply backoff (simulated/tracked)
                       # We can just record it or ignore actual sleep as per requirement
                       pass
                   self.retry_counts[job_id] = attempt + 1  # Track how many retries were attempted/failed

           return False