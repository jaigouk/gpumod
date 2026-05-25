from __future__ import annotations
   import time
   import uuid
   from collections import deque
   from dataclasses import dataclass, field
   from typing import Any, Optional

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       error: Optional[str] = None

       def mark_processed(self):
           self.status = "completed"

       def mark_failed(self, error: str):
           self.status = "failed"
           self.error = error

   class JobQueue:
       def __init__(self):
           self._queue: deque[Job] = deque()

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

       def __iter__(self):
           return iter(self._queue)