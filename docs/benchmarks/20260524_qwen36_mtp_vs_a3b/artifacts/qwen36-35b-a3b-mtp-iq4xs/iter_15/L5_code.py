from __future__ import annotations
   from dataclasses import dataclass, field
   from typing import Any, Optional
   import time
   from collections import deque

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       result: Optional[Any] = None

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           return self._queue.popleft() if self._queue else None

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def __len__(self) -> int:
           return len(self._queue)