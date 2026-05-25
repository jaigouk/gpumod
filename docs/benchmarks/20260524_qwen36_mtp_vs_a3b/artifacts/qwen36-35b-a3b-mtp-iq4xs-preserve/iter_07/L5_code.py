import time
   from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   from queue import Queue as StdQueue

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       retry_count: int = 0
       max_retries: int = 3
       created_at: float = field(default_factory=time.time)

   class JobQueue:
       def __init__(self):
           self._queue = StdQueue()

       def enqueue(self, job: Job) -> None:
           self._queue.put(job)

       def dequeue(self) -> Optional[Job]:
           if self._queue.empty():
               return None
           return self._queue.get()

       def is_empty(self) -> bool:
           return self._queue.empty()

       def size(self) -> int:
           return self._queue.qsize()

       def process(self, handler: Callable[[Job], Any]) -> None:
           while not self.is_empty():
               job = self.dequeue()
               try:
                   handler(job)
                   job.status = "completed"
               except Exception as e:
                   job.status = "failed"
                   raise e