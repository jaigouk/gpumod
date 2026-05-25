from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, List, Optional

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.now)
       retry_count: int = 0

   class JobQueue:
       def __init__(self):
           self._queue: List[Job] = []

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