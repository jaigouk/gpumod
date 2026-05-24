def process_job(self, job_id: str, processor: Callable) -> bool:
       if job_id not in self.jobs:
           return False
       job = self.jobs[job_id]
       max_retries = 3
       for attempt in range(max_retries + 1):  # 0 to 3 => 4 attempts total? Wait.
           # Requirement: "retry up to 3 times"
           # Usually means: 1 initial attempt + 3 retries = 4 total, or 3 total attempts?
           # Let's interpret "retry up to 3 times" as: if it fails, retry 3 times. So total attempts = 1 + 3 = 4.
           # But the backoff sequence is 1s, 2s, 4s. That's 3 delays, which matches 3 retries.
           # So: attempt 0 (initial) -> fail -> wait 1s (retry 1) -> fail -> wait 2s (retry 2) -> fail -> wait 4s (retry 3) -> fail -> return False.
           # I'll stick to: initial attempt + up to 3 retries.
           try:
               processor(job["data"])
               return True
           except Exception:
               if attempt < max_retries:
                   delay = 2 ** attempt  # 1, 2, 4
                   job["backoff_delays"].append(delay)
                   job["retry_count"] = attempt + 1
                   # Simulate sleep by just recording, or actually sleep? Requirement says "can be simulated"
                   # I'll just record it. If they want actual sleep, I'd add time.sleep(delay), but requirement says stored/tracked rather than actually sleeping. I'll skip time.sleep.
               else:
                   return False