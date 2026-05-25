from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   from collections import deque
   import uuid

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue = deque()

       def add(self, job: Job) -> None:
           self._queue.append(job)

       def get_next(self) -> Optional[Job]:
           if self._queue:
               job = self._queue.popleft()
               job.status = "processing"
               return job
           return None

       def complete(self, job: Job) -> None:
           job.status = "completed"

       def fail(self, job: Job) -> None:
           job.status = "failed"