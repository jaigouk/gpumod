from dataclasses import dataclass, field
   from typing import Any, Optional
   from collections import deque
   import time
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0  # Lower number = higher priority
       status: str = "pending"
       created_at: float = field(default_factory=time.time)
       attempts: int = 0

   class JobQueue:
       def __init__(self):
           self._queue: deque[Job] = deque()
           self._lock = None  # Could add threading lock later if needed, but keep it simple

       def enqueue(self, job: Job) -> None:
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           return self._queue.popleft() if self._queue else None

       def peek(self) -> Optional[Job]:
           return self._queue[0] if self._queue else None

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0