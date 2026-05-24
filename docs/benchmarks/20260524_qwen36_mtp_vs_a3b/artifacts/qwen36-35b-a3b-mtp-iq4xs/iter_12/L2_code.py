def process_job(self, job_id: str, processor: Callable) -> bool:
       if job_id not in self.jobs:
           return False
       job = self.jobs[job_id]
       max_retries = 3
       base_delay = 1
       for attempt in range(max_retries + 1):
           try:
               processor(job["data"])
               return True
           except Exception:
               if attempt < max_retries:
                   delay = base_delay * (2 ** attempt)
                   job["backoff_delays"].append(delay)
                   job["retry_count"] += 1
               else:
                   return False