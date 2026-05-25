import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Dict[str, Any]) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0  # seconds

           for attempt in range(max_retries + 1):  # 0 to 3 (4 total attempts? Wait, requirement says "retry up to 3 times")
               # Let's clarify: "retry up to 3 times" usually means 1 initial attempt + 3 retries = 4 total, or 1 initial + 3 retries?
               # Actually, "retry up to 3 times" typically means if it fails, try again up to 3 times. So total attempts = 4.
               # But backoff delays are 1s, 2s, 4s. That's 3 delays. So initial attempt + 3 retries = 4 attempts total.
               # Let's stick to: attempt 0 (initial), then retries 1, 2, 3 with delays 1, 2, 4.
               # Wait, the example says "retry up to 3 times" and backoff is 1s, 2s, 4s. That's exactly 3 retries.
               # So loop range(4) or handle retries explicitly.

               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception as e:
                   if attempt < max_retries:
                       delay = base_delay * (2 ** attempt)
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate sleep
                       # time.sleep(delay) # Requirement says can be simulated/stored
                   else:
                       return False