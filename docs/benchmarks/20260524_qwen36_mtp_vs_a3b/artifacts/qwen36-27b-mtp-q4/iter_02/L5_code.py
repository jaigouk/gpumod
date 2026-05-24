from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import uuid
   from collections import deque

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       func: Callable = field(repr=False)
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       priority: int = 0
       status: str = "pending"
       result: Any = None
       error: Optional[Exception] = None

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if self._queue:
               return self._queue.popleft()
           return None

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0