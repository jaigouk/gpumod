import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, float] = {}  # To track simulated backoff

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = 0.0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           backoff_base = 1.0  # seconds

           for attempt in range(max_retries + 1):  # 0 to 3 (1 initial + 3 retries)
               try:
                   data = self.jobs.get(job_id)
                   processor(data)
                   # Success
                   self.retry_counts[job_id] = attempt
                   return True
               except Exception:
                   if attempt < max_retries:
                       # Calculate backoff delay: 1, 2, 4 for retries 1, 2, 3
                       delay = backoff_base * (2 ** attempt)
                       self.backoff_delays[job_id] = delay
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate sleep by just tracking delay instead of time.sleep()
                   else:
                       # All retries exhausted
                       self.retry_counts[job_id] = max_retries
                       return False
           # Fallback (shouldn't reach here, but for type safety)
           return False