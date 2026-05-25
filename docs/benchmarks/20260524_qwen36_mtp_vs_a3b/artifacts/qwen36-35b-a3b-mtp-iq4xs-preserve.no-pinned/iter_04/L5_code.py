from __future__ import annotations
   import time
   from dataclasses import dataclass, field
   from typing import Any, Optional
   from collections import deque

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       priority: int = 0

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue.popleft()

       def peek(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue[0]

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)