from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import time
   import threading
   from queue import Queue as StdQueue

   @dataclass
   class Job:
       id: str
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       status: str = "pending"
       result: Any = None
       error: Optional[Exception] = None
       created_at: float = field(default_factory=time.time)

   class JobQueue:
       def __init__(self):
           self._queue = StdQueue()
           self._lock = threading.Lock()

       def enqueue(self, job: Job) -> None:
           with self._lock:
               self._queue.put(job)

       def dequeue(self) -> Optional[Job]:
           with self._lock:
               if self._queue.empty():
                   return None
               return self._queue.get()

       def size(self) -> int:
           with self._lock:
               return self._queue.qsize()

       def is_empty(self) -> bool:
           return self.size() == 0