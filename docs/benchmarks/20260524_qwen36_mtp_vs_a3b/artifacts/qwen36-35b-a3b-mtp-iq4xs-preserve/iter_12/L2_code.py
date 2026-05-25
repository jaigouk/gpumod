def process_job(self, job_id: str, processor: Callable) -> bool:
       if job_id not in self.jobs:
           return False

       job = self.jobs[job_id]
       max_retries = 3
       backoff_delays = [1, 2, 4]

       # First attempt
       try:
           processor(job["data"])
           return True
       except Exception:
           pass

       # Retries
       for i in range(max_retries):
           job["retry_count"] += 1
           delay = backoff_delays[i]
           job["backoff_delays"].append(delay)
           # Simulate backoff without actually sleeping
           # Could store it or just continue
           try:
               processor(job["data"])
               return True
           except Exception:
               continue

       return False