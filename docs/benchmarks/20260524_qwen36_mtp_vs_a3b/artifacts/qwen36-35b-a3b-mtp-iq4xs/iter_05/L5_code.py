from dataclasses import dataclass, field
   from typing import Callable, Any, List, Optional
   import time

   @dataclass
   class Job:
       id: str
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       status: str = "pending"
       priority: int = 0
       attempts: int = 0

   class JobQueue:
       def __init__(self):
           self.jobs: List[Job] = []

       def add(self, job: Job):
           self.jobs.append(job)

       def get_next(self) -> Optional[Job]:
           if not self.jobs:
               return None
           # Simple FIFO for base class
           return self.jobs.pop(0)

       def process(self, job: Job) -> Any:
           job.status = "running"
           job.attempts += 1
           try:
               result = job.func(*job.args, **job.kwargs)
               job.status = "completed"
               return result
           except Exception as e:
               job.status = "failed"
               raise e