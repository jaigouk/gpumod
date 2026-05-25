from __future__ import annotations
   import dataclasses
   from collections import deque
   from datetime import datetime, timezone
   from typing import Any, Callable, Optional

   @dataclasses.dataclass
   class Job:
       id: str
       func: Callable
       args: tuple = ()
       kwargs: dict = None
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3
       created_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))

       def __post_init__(self):
           if self.kwargs is None:
               self.kwargs = {}

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           return self._queue.popleft() if self._queue else None

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)

       def clear(self) -> None:
           self._queue.clear()