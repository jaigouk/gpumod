from dataclasses import dataclass, field
   from typing import Any, List, Optional
   import time

   @dataclass
   class Job:
       id: str
       payload: Any
       priority: int = 0
       created_at: float = field(default_factory=time.time)
       attempts: int = 0

   class JobQueue:
       def __init__(self):
           self._queue: List[Job] = []

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           return self._queue.pop(0) if self._queue else None

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)