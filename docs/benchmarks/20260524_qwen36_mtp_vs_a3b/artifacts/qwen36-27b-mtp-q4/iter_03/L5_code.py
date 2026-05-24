import uuid
   import time
   from dataclasses import dataclass, field
   from typing import Any, Optional
   from queue import Queue
   from enum import Enum

   class JobStatus(Enum):
       PENDING = "pending"
       RUNNING = "running"
       COMPLETED = "completed"
       FAILED = "failed"

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       status: JobStatus = JobStatus.PENDING
       created_at: float = field(default_factory=time.time)
       priority: int = 0  # For priority queue compatibility
       retries: int = 0

   class JobQueue:
       def __init__(self, maxsize: int = 0):
           self._queue = Queue(maxsize=maxsize)

       def enqueue(self, job: Job) -> None:
           self._queue.put(job)

       def dequeue(self, timeout: Optional[float] = None) -> Optional[Job]:
           try:
               return self._queue.get(timeout=timeout)
           except Exception:
               return None

       def size(self) -> int:
           return self._queue.qsize()

       def is_empty(self) -> bool:
           return self.size() == 0