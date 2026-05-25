import time
   from typing import Callable, Any, Dict, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Dict[str, Any]] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, float] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = 0.0

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           base_delay = 1.0
           current_delay = base_delay

           for attempt in range(max_retries + 1):  # 0 to 3 (4 attempts total? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 total, or just 3 attempts total. Let's clarify: "retry up to 3 times" typically means if it fails, retry 1, 2, 3 times. So total attempts = 4. But sometimes it means max 3 attempts total. I'll stick to 3 retries = 4 attempts, or just 3 attempts total. Let's re-read: "retry up to 3 times". I'll interpret as max 3 retries (4 attempts). Actually, standard is often 3 attempts total. Let's do 3 retries (4 attempts) to be safe, or just 3 attempts total. I'll use `range(4)` for attempts, with backoff before retries 1, 2, 3. Wait, backoff sequence: 1s, 2s, 4s. That's 3 delays. So 3 retries.
           # Let's structure it carefully:
           # attempt 0: call processor
           # if fails, delay 1s, retry 1
           # if fails, delay 2s, retry 2
           # if fails, delay 4s, retry 3
           # if fails, return False
           # So max_retries = 3
           # delays = [1, 2, 4]

           # I'll track retry count in self.retry_counts[job_id]
           # I'll store backoff delays in self.backoff_delays[job_id] or just track them.
           # The requirement says: "The backoff delays can be stored/tracked rather than actually sleeping."
           # So I'll store the delays in the job's state or a separate dict.

           # Let's refine:
           attempts = 0
           max_retries = 3
           delays = [1.0, 2.0, 4.0]

           while attempts <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   # Success
                   self.retry_counts[job_id] = attempts
                   return True
               except Exception:
                   if attempts < max_retries:
                       self.retry_counts[job_id] = attempts + 1
                       # Store delay instead of sleeping
                       self.backoff_delays[job_id] = delays[attempts]
                       # Simulate sleep or just track
                       # time.sleep(delays[attempts]) # Requirement says can be stored/tracked rather than actually sleeping
                   else:
                       self.retry_counts[job_id] = attempts
                       return False
               attempts += 1
           return False