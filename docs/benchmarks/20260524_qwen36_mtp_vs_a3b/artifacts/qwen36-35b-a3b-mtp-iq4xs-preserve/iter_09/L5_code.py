from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time

   @dataclass
   class Job:
       id: str
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       created_at: float = field(default_factory=time.time)
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._processed = []

       def add(self, job: Job) -> None:
           self._queue.append(job)

       def get_next(self) -> Optional[Job]:
           if not self._queue:
               return None
           job = self._queue.pop(0)
           job.status = "running"
           return job

       def complete(self, job: Job) -> None:
           job.status = "completed"
           self._processed.append(job)

       def clear(self) -> None:
           self._queue.clear()
           self._processed.clear()