from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time

   @dataclass
   class Job:
       name: str
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       created_at: float = field(default_factory=time.time)
       status: str = "pending"  # pending, running, completed, failed

   class JobQueue:
       def __init__(self):
           self._jobs: list[Job] = []

       def enqueue(self, job: Job):
           self._jobs.append(job)

       def dequeue(self) -> Optional[Job]:
           if not self._jobs:
               return None
           return self._jobs.pop(0)

       def is_empty(self) -> bool:
           return len(self._jobs) == 0

       def process(self, job: Job):
           job.status = "running"
           try:
               job.func(*job.args, **job.kwargs)
               job.status = "completed"
           except Exception as e:
               job.status = "failed"
               raise e