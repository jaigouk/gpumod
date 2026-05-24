from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time

   @dataclass
   class Job:
       job_id: str
       payload: Any
       created_at: float = field(default_factory=time.time)
       status: str = "pending"
       result: Optional[Any] = None
       error: Optional[Exception] = None

   class JobQueue:
       def __init__(self):
           self._queue = []
           self._current_id = 0

       def enqueue(self, job: Job) -> str:
           self._queue.append(job)
           return job.job_id

       def dequeue(self) -> Optional[Job]:
           if not self._queue:
               return None
           return self._queue.pop(0)

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0