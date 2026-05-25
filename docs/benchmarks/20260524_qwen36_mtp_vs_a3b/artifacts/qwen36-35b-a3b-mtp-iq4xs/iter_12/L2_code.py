def process_job(self, job_id: str, processor: Callable) -> bool:
       if job_id not in self.jobs:
           raise ValueError(f"Job {job_id} not found")

       max_retries = 3
       backoff_delays = [1, 2, 4]  # in seconds
       attempts = 0

       while attempts <= max_retries:
           try:
               processor(self.jobs[job_id])
               return True
           except Exception:
               if attempts < max_retries:
                   # Track retry count and backoff
                   self.retry_counts[job_id] = attempts + 1
                   # Simulate backoff by just recording it or doing nothing
                   # The prompt says "backoff delays can be stored/tracked rather than actually sleeping"
                   # I'll just record the delay or compute it. Let's store it in a separate dict or just note it.
                   # Actually, I'll just use a list to track backoff per job if needed, or just compute it.
                   # Simpler: just record the delay in the retry_counts or a separate dict.
                   # Let's add a `backoff_delays` dict to track per job.
                   pass
               else:
                   return False
           attempts += 1