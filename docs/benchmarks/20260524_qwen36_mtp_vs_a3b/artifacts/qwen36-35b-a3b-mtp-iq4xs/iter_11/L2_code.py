def process_job(self, job_id: str, processor: Callable) -> bool:
       max_retries = 3
       backoff_times = [1, 2, 4]
       retry_count = 0
       self.retry_counts[job_id] = 0
       self.backoff_delays[job_id] = []

       for attempt in range(max_retries):
           try:
               processor(self.jobs[job_id])
               self.retry_counts[job_id] = attempt
               return True
           except Exception as e:
               self.retry_counts[job_id] = attempt + 1
               if attempt < max_retries - 1:
                   delay = backoff_times[attempt]
                   self.backoff_delays[job_id].append(delay)
                   # Simulate backoff instead of sleeping
                   # time.sleep(delay)
                   pass
       return False