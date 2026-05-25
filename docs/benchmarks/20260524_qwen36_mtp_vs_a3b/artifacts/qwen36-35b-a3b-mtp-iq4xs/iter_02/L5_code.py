from __future__ import annotations
   import time
   from dataclasses import dataclass, field
   from typing import Any, Callable, Optional

   from .priority import PriorityQueue

   @dataclass
   class Job:
       id: str
       payload: Any
       priority: int = 0
       status: str = "pending"
       attempts: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self, max_retries: int = 3):
           self._queue = PriorityQueue()
           self.max_retries = max_retries

       def add_job(self, job: Job) -> None:
           self._queue.push(job)

       def process_next(self, handler: Callable[[Job], Any]) -> Optional[Any]:
           job = self._queue.pop()
           if job is None:
               return None
           return handler(job)

       def size(self) -> int:
           return self._queue.size()