max_retries = 3
   backoff_delays = [1, 2, 4]
   for attempt in range(max_retries + 1):
       try:
           processor(job_data)
           return True
       except Exception:
           self.retry_counts[job_id] = attempt + 1
           if attempt < max_retries:
               # Track backoff delay
               self.backoff_delays[job_id] = self.backoff_delays.get(job_id, []) + [backoff_delays[attempt]]
               # Simulate backoff (optional, but prompt says can be stored/tracked)
               # I'll just record it. If actual sleep is needed, I'll skip it per prompt.
           else:
               return False