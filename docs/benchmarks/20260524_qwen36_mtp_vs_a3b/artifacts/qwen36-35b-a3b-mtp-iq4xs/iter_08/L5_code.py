from __future__ import annotations
   import time
   from dataclasses import dataclass, field
   from collections import deque
   from typing import Any, Optional

   @dataclass
   class Job:
       id: str
       name: str
       payload: Any = None
       status: str = "pending"  # pending, processing, completed, failed
       created_at: float = field(default_factory=time.time)
       updated_at: float = field(default_factory=time.time)

   class JobQueue:
       def __init__(self):
           self._queue: deque[Job] = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self.is_empty():
               return None
           job = self._queue.popleft()
           job.status = "processing"
           job.updated_at = time.time()
           return job

       def peek(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue[0]

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)