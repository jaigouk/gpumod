from dataclasses import dataclass, field
   from typing import Any, Callable, List, Optional

   @dataclass
   class Job:
       func: Callable
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       job_id: str = field(default_factory=lambda: "job-0")
       priority: int = 0
       max_retries: int = 3
       attempts: int = 0

   class JobQueue:
       def __init__(self):
           self._queue: List[Job] = []

       def add(self, job: Job) -> None:
           self._queue.append(job)

       def get_next(self) -> Optional[Job]:
           if self.is_empty():
               return None
           return self._queue.pop(0)

       def is_empty(self) -> bool:
           return len(self._queue) == 0

       def size(self) -> int:
           return len(self._queue)