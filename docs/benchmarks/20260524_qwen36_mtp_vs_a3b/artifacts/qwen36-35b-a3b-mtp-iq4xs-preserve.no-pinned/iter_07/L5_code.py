from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time

   @dataclass
   class Job:
       job_id: str
       task: Callable[..., Any]
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       result: Any = None
       error: Optional[Exception] = None

   class JobQueue:
       def __init__(self):
           self._queue = []

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)
           job.status = "queued"

       def dequeue(self) -> Optional[Job]:
           if self._queue:
               job = self._queue.pop(0)
               job.status = "running"
               return job
           return None

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0