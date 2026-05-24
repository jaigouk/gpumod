from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Callable, Optional
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       task: Callable[..., Any] = None
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       priority: int = 0
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3
       created_at: datetime = field(default_factory=datetime.now)

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