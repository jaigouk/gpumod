from dataclasses import dataclass, field
   from typing import Any, Callable, List, Optional
   import time

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       attempts: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue: List[Job] = []

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self._queue:
               return self._queue.pop(0)
           return None

       def process(self, handler: Callable[[Job], Any]) -> Any:
           job = self.dequeue()
           if job:
               job.status = "processing"
               result = handler(job)
               job.status = "completed"
               return result
           return None