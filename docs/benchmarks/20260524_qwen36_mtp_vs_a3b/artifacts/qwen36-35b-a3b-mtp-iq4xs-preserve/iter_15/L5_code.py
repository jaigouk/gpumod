from dataclasses import dataclass, field
   from typing import Any, Optional, List
   import time
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       status: str = "pending"  # pending, processing, completed, failed
       priority: int = 0
       created_at: float = field(default_factory=time.time)
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

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)