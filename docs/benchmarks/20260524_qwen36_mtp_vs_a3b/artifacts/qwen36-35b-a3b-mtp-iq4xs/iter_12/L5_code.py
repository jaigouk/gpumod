from __future__ import annotations
   import uuid
   from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   from collections import deque

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       status: str = "pending"
       retries: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue: deque[Job] = deque()

       def add(self, job: Job) -> None:
           self._queue.append(job)

       def get_next(self) -> Optional[Job]:
           if not self._queue:
               return None
           return self._queue.popleft()

       def process(self, job: Job, handler: Callable[[Any], Any]) -> Any:
           if job.status != "pending":
               raise ValueError(f"Job {job.id} is not pending (status: {job.status})")
           try:
               result = handler(job.payload)
               job.status = "completed"
               return result
           except Exception as e:
               job.status = "failed"
               raise e

       def is_empty(self) -> bool:
           return len(self._queue) == 0