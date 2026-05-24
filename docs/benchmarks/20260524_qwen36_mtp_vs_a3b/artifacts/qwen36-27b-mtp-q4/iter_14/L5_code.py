from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Optional
   from collections import deque
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       data: Any = None
       created_at: datetime = field(default_factory=datetime.now)
       status: str = "pending"
       priority: int = 0  # Lower number = higher priority

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def enqueue(self, job: Job) -> None:
           job.status = "queued"
           self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           if not self._queue:
               return None
           job = self._queue.popleft()
           job.status = "processing"
           return job

       def size(self) -> int:
           return len(self._queue)

       def is_empty(self) -> bool:
           return len(self._queue) == 0