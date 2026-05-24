from dataclasses import dataclass, field
   from datetime import datetime
   from typing import Any, Callable, Optional
   import uuid
   import queue

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       payload: Any = None
       priority: int = 0  # Lower number = higher priority
       created_at: datetime = field(default_factory=datetime.now)
       status: str = "pending"  # pending, processing, completed, failed
       retries: int = 0
       max_retries: int = 3

   class JobQueue:
       def __init__(self):
           self._queue = queue.Queue()

       def enqueue(self, job: Job) -> None:
           self._queue.put(job)

       def dequeue(self) -> Optional[Job]:
           try:
               return self._queue.get_nowait()
           except queue.Empty:
               return None

       def process(self, job: Job, handler: Callable[[Job], Any]) -> None:
           job.status = "processing"
           try:
               handler(job)
               job.status = "completed"
           except Exception as e:
               job.status = "failed"
               raise e

       def __len__(self) -> int:
           return self._queue.qsize()