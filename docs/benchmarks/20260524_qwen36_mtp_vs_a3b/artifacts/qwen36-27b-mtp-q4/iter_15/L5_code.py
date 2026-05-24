from dataclasses import dataclass, field
   from typing import Any, Optional
   import uuid
   from datetime import datetime

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.now)
       attempts: int = 0

   class JobQueue:
       def __init__(self):
           self._queue: list[Job] = []

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if not self._queue:
               return None
           return self._queue.pop(0)

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0