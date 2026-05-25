from dataclasses import dataclass, field
   from typing import Any, Optional
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0
       retries: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue = []

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue.pop(0)

       def peek(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue[0]

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)