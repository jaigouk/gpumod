import uuid
   import time
   from dataclasses import dataclass, field
   from typing import Callable, Any, Optional
   from queue import Queue

   @dataclass
   class Job:
       id: str = field(default_factory=lambda: str(uuid.uuid4()))
       task: Callable = None
       args: tuple = ()
       kwargs: dict = field(default_factory=dict)
       priority: int = 0
       created_at: float = field(default_factory=time.time)
       status: str = "pending"
       result: Any = None
       error: Optional[str] = None

   class JobQueue:
       def __init__(self, maxsize: int = 0):
           self._queue = Queue(maxsize=maxsize)
           self._jobs = {}  # Track jobs by ID

       def add(self, job: Job) -> str:
           job.status = "pending"
           self._queue.put(job)
           self._jobs[job.id] = job
           return job.id

       def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Job]:
           try:
               job = self._queue.get(block=block, timeout=timeout)
               job.status = "running"
               return job
           except Exception:
               return None

       def task_done(self):
           self._queue.task_done()

       def complete(self, job: Job, result: Any = None):
           job.status = "completed"
           job.result = result
           self.task_done()

       def fail(self, job: Job, error: str):
           job.status = "failed"
           job.error = error
           self.task_done()