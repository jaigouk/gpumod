import time
   from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import queue

   @dataclass
   class Job:
       job_id: str
       payload: Any
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       retries: int = 0

   class JobQueue:
       def __init__(self):
           self._queue = queue.Queue()

       def enqueue(self, job: Job) -> None:
           self._queue.put(job)

       def dequeue(self) -> Optional[Job]:
           try:
               return self._queue.get_nowait()
           except queue.Empty:
               return None

       def is_empty(self) -> bool:
           return self._queue.empty()

       def size(self) -> int:
           return self._queue.qsize()