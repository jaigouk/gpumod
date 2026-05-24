from dataclasses import dataclass, field
   from typing import Any, Callable, Optional
   import threading
   import uuid
   from collections import deque

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       task: Callable[..., Any] = None
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       priority: int = 0
       max_retries: int = 3
       retries: int = 0
       status: str = "pending"

   class JobQueue:
       def __init__(self):
           self._queue = deque()
           self._lock = threading.Lock()

       def enqueue(self, job: Job) -> None:
           with self._lock:
               self._queue.append(job)

       def dequeue(self) -> Optional[Job]:
           with self._lock:
               return self._queue.popleft() if self._queue else None

       def is_empty(self) -> bool:
           with self._lock:
               return len(self._queue) == 0

       def size(self) -> int:
           with self._lock:
               return len(self._queue)