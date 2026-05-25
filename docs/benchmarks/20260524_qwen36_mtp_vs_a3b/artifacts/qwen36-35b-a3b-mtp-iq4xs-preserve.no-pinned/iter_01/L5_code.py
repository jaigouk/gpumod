from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Optional
   import queue

   @dataclass
   class Job:
       id: str
       payload: Any
       priority: int = 0
       status: str = "pending"
       retry_count: int = 0
       created_at: datetime = field(default_factory=datetime.utcnow)
       last_updated: datetime = field(default_factory=datetime.utcnow)

   class JobQueue:
       def __init__(self):
           self._queue = queue.Queue()

       def add(self, job: Job):
           self._queue.put(job)

       def get(self) -> Optional[Job]:
           try:
               return self._queue.get_nowait()
           except queue.Empty:
               return None

       def is_empty(self) -> bool:
           return self._queue.empty()

       def size(self) -> int:
           return self._queue.qsize()