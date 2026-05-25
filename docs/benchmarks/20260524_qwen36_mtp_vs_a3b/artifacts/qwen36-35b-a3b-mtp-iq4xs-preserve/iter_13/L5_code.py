from dataclasses import dataclass, field
   from datetime import datetime, timezone
   from typing import Any, Optional, List
   from collections import deque

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
       attempts: int = 0

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

       def __len__(self) -> int:
           return len(self._queue)