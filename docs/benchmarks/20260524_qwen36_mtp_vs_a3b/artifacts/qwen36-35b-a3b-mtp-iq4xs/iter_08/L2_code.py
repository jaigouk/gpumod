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
           if job_id not in self.jobs:
               raise ValueError(f"Job {job_id} not found")

           max_retries = 3
           base_delay = 1.0  # seconds

           # Attempt initial + retries
           for attempt in range(max_retries + 1):
               try:
                   processor(self.jobs[job_id])
                   self.retry_counts[job_id] = attempt  # or keep it as number of retries
                   return True
               except Exception as e:
                   if attempt < max_retries:
                       # Calculate backoff: 1, 2, 4 for retries 0, 1, 2
                       delay = base_delay * (2 ** attempt)
                       self.backoff_delays[job_id].append(delay)
                       self.retry_counts[job_id] = attempt + 1
                       # Simulate sleep by just recording, or actually sleep?
                       # Prompt says "can be stored/tracked rather than actually sleeping"
                       # I'll just record it. If they want actual sleep, I can add time.sleep(delay)
                       # But it says "can be stored/tracked rather than actually sleeping", so I'll skip time.sleep or make it optional.
                       # I'll just track it.
                   else:
                       return False