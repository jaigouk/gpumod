from typing import Callable, Dict, Any, List

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, List[float]] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           if job_id not in self.jobs:
               return False

           # Reset retry count and delays for this job when processing?
           # The requirement says "Track retry count for each job". It might be better to reset on each process_job call, or keep it persistent.
           # I'll reset it when process_job is called to match typical usage, or keep it as instance state.
           # Actually, the example shows process_job being called once. I'll reset retry count and delays at the start of process_job.
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

           max_retries = 3
           base_delay = 1.0

           while self.retry_counts[job_id] <= max_retries:
               try:
                   processor(self.jobs[job_id])
                   return True
               except Exception:
                   self.retry_counts[job_id] += 1
                   if self.retry_counts[job_id] <= max_retries:
                       delay = base_delay * (2 ** (self.retry_counts[job_id] - 1))
                       self.backoff_delays[job_id].append(delay)
                   else:
                       return False
           return False