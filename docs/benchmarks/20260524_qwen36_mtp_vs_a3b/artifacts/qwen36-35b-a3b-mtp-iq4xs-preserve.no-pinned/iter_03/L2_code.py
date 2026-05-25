max_retries = 3
   for attempt in range(max_retries + 1):
       try:
           processor(self.jobs[job_id]["data"])
           return True
       except Exception:
           if attempt < max_retries:
               self.retry_counts[job_id] = attempt + 1
               backoff = 2 ** attempt  # 1, 2, 4 for attempts 0, 1, 2
               self.jobs[job_id]["backoff_delays"] = self.jobs[job_id].get("backoff_delays", [])
               self.jobs[job_id]["backoff_delays"].append(backoff)
           else:
               return False