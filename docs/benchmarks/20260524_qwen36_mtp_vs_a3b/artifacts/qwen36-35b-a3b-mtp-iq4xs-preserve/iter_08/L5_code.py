from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Callable, Optional
   from queue import Queue as _BasicQueue

   @dataclass
   class Job:
       id: str
       payload: Any
       status: str = "pending"
       created_at: datetime = field(default_factory=datetime.utcnow)

   class JobQueue:
       def __init__(self):
           self._queue = _BasicQueue()

       def enqueue(self, job: Job) -> None:
           self._queue.put(job)

       def dequeue(self) -> Optional[Job]:
           if self._queue.empty():
               return None
           return self._queue.get()

       def is_empty(self) -> bool:
           return self._queue.empty()

       def size(self) -> int:
           return self._queue.qsize()