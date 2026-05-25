from dataclasses import dataclass, field
   from typing import Any, Callable, List, Optional
   import time

   @dataclass
   class Job:
       job_id: str
       payload: Any
       status: str = "pending"
       attempts: int = 0
       max_retries: int = 3
       created_at: float = field(default_factory=time.time)

   class JobQueue:
       def __init__(self):
           self._queue: List[Job] = []

       def add(self, job: Job) -> None:
           self._queue.append(job)

       def pop(self) -> Optional[Job]:
           if not self._queue:
               return None
           return self._queue.pop(0)

       def __len__(self) -> int:
           return len(self._queue)