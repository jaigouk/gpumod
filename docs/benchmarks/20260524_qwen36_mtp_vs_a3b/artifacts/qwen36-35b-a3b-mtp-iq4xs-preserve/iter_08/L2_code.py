def process_job(self, job_id: str, processor: Callable) -> bool:
       if job_id not in self.jobs:
           return False

       retry_count = 0
       max_retries = 3
       backoff_delays = []

       while retry_count <= max_retries:
           try:
               job_data = self.jobs[job_id]["data"]
               processor(job_data)
               self.jobs[job_id]["retry_count"] = retry_count
               self.jobs[job_id]["backoff_delays"] = backoff_delays
               return True
           except Exception:
               if retry_count < max_retries:
                   backoff_delays.append(2 ** retry_count) # 1, 2, 4
                   retry_count += 1
                   # Simulate backoff by storing it
               else:
                   self.jobs[job_id]["retry_count"] = retry_count
                   self.jobs[job_id]["backoff_delays"] = backoff_delays
                   return False
       return False