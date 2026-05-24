import time
   from typing import Callable, Any, Dict

   class JobQueue:
       def __init__(self):
           self.jobs: Dict[str, Any] = {}
           self.retry_counts: Dict[str, int] = {}
           self.backoff_delays: Dict[str, list] = {}

       def add_job(self, job_id: str, data: Any) -> None:
           self.jobs[job_id] = data
           self.retry_counts[job_id] = 0
           self.backoff_delays[job_id] = []

       def process_job(self, job_id: str, processor: Callable) -> bool:
           # implementation here