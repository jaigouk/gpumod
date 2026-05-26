import time

   class JobQueue:
       def __init__(self):
           self.jobs = {}
           self.max_retries = 3
           self.backoff_delays = [1, 2, 4]

       def add_job(self, job_id, data):
           self.jobs[job_id] = {
               "data": data,
               "retry_count": 0,
               "status": "pending",
               "next_backoff": 0
           }

       def process_job(self, job_id, processor):
           if job_id not in self.jobs:
               return False

           job = self.jobs[job_id]
           retry_count = job["retry_count"]

           for attempt in range(self.max_retries + 1):  # 0 to 3 (1st try + 3 retries)
               try:
                   processor(job["data"])
                   job["status"] = "completed"
                   return True
               except Exception:
                   retry_count += 1
                   job["retry_count"] = retry_count
                   if retry_count <= self.max_retries:
                       # Determine backoff delay
                       idx = retry_count - 1
                       delay = self.backoff_delays[idx] if idx < len(self.backoff_delays) else 2 ** retry_count
                       job["next_backoff"] = delay
                       # Simulated sleep
                       # time.sleep(delay)
                   else:
                       job["status"] = "failed"
                       return False
           
           job["status"] = "failed"
           return False