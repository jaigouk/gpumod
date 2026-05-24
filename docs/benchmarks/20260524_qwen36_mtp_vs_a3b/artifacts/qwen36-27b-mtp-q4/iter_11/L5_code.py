from dataclasses import dataclass, field
   from typing import Any, Optional
   import time
   import threading

   @dataclass
   class Job:
       id: str
       data: Any
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       retry_count: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue: list[Job] = []
           self._lock = threading.Lock()

       def enqueue(self, job: Job) -> None:
           with self._lock:
               self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           with self._lock:
               return self._queue.pop(0) if self._queue else None

       def size(self) -> int:
           with self._lock:
               return len(self._queue)

       def is_empty(self) -> bool:
           return self.size() == 0