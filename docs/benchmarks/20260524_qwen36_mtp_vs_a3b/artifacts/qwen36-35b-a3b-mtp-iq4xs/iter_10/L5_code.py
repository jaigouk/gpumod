from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time

   @dataclass
   class Job:
       job_id: str
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       created_at: float = field(default_factory=time.time)
       status: str = "pending"  # pending, running, completed, failed

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._completed = []
           self._failed = []

       def enqueue(self, job: Job):
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if not self._queue:
               return None
           return self._queue.pop(0)

       def get_jobs(self, status: Optional[str] = None):
           if status:
               return [j for j in self._queue + self._completed + self._failed if j.status == status]
           return self._queue + self._completed + self._failed