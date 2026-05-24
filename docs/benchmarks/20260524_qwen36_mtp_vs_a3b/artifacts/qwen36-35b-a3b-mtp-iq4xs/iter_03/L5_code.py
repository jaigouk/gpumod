from dataclasses import dataclass, field
   from typing import Any, Callable, List, Optional
   import time
   import threading

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
           self._lock = threading.Lock()

       def add(self, job: Job) -> None:
           with self._lock:
               self._queue.append(job)

       def pop(self) -> Optional[Job]:
           with self._lock:
               if self._queue:
                   return self._queue.pop(0)
               return None

       def size(self) -> int:
           with self._lock:
               return len(self._queue)

       def process(self, job: Job, handler: Callable[[Job], Any]) -> Any:
           with self._lock:
               job.status = "processing"
           try:
               result = handler(job)
               with self._lock:
                   job.status = "completed"
               return result
           except Exception as e:
               with self._lock:
                   job.status = "failed"
               raise e