import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           base_delay = 1.0

           for attempt in range(max_retries + 1):  # 0 to 3 -> 4 attempts total? Wait, requirement says "retry up to 3 times". Usually means 1 initial + 3 retries = 4 attempts, or just 3 attempts total. Let's interpret "retry up to 3 times" as max_retries = 3, so total attempts = 4. But often in these prompts, they mean 3 attempts total. I'll stick with max_retries = 3 (so 1 initial + 3 retries = 4 attempts). Actually, the backoff sequence is 1s, 2s, 4s. That's 3 delays. So if it fails on attempt 1, wait 1s, retry. Fails on attempt 2, wait 2s, retry. Fails on attempt 3, wait 4s, retry. If fails on attempt 4, return False. That matches 3 retries.
           # Let's adjust: attempts = 0 to 3 (4 total). Delays: after attempt 0 -> 1s, after 1 -> 2s, after 2 -> 4s. After 3 -> exhausted.
           # Actually, simpler: for attempt in range(4): ... if success: return True. delay = 2**attempt. But delays are 1, 2, 4. So delay = 2**attempt works for attempt 0,1,2.
           # Let's track retry count explicitly as required: "Track retry count for each job"
           # I'll increment retry count on failure.