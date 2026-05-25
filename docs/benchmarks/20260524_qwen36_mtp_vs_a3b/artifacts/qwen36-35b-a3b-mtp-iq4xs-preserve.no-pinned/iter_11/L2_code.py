import time
   from typing import Callable, Any, Dict, List, Optional

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.retry_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.retry_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           max_retries = 3
           base_delay = 1.0
           success = False

           for attempt in range(max_retries + 1):  # 0, 1, 2, 3 -> total 4 attempts? Wait, "retry up to 3 times" usually means 1 initial + 3 retries = 4 attempts total, or initial + 3 retries. Let's clarify: "retry up to 3 times" typically means if it fails, try again up to 3 times. So attempts = 1 (initial) + 3 (retries) = 4. But sometimes it means total attempts = 3. I'll interpret as: initial attempt + up to 3 retries = 4 attempts total. However, the backoff sequence is 1s, 2s, 4s. That's 3 delays, which matches 3 retries. So: attempt 0 -> fail -> wait 1s -> attempt 1 -> fail -> wait 2s -> attempt 2 -> fail -> wait 4s -> attempt 3 -> fail -> return False.
           # Let's structure it carefully.
           # Actually, standard retry logic:
           # for i in range(max_retries + 1):
           #   try: ... except:
           #     if i == max_retries: return False
           #     delay = base_delay * (2 ** i)
           #     track delay
           #     time.sleep(delay) # or just track
           # return True

           # Wait, the prompt says "exponential backoff between retries: 1s, 2s, 4s". That's exactly 3 delays.
           # So max_retries = 3.
           # I'll track the delays in the class state.

           self.retry_counts[job_id] = 0
           self.retry_delays[job_id] = []

           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   if attempt == max_retries:
                       return False
                   delay = base_delay * (2 ** attempt)
                   self.retry_counts[job_id] = attempt + 1
                   self.retry_delays[job_id].append(delay)
                   # Simulate sleep by just tracking, or actually sleep? Prompt says "can be stored/tracked rather than actually sleeping." I'll just track it, but to be safe, I'll use time.sleep but note it's optional. Actually, I'll just calculate and store, but I'll add a flag or just skip sleep to follow "can be stored/tracked rather than actually sleeping". I'll just store it and not sleep, or I'll sleep but the prompt says it *can* be tracked. I'll just track it.

           return False