class JobQueue:
       def __init__(self):
           self.jobs = {}
           self.retry_counts = {}
           self.backoff_delays = {}

       def add_job(self, job_id: str, data: dict):
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           max_retries = 3
           base_delay = 1.0
           attempt = 0

           while attempt < max_retries:
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   attempt += 1
                   if attempt <= max_retries:
                       delay = base_delay * (2 ** (attempt - 1))
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt
                       if attempt == max_retries:
                           return False